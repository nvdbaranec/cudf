/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "nvcomp_decode.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <io/utilities/hostdevice_vector.hpp>

#include <thrift/protocol/TCompactProtocol.h>
#include <thrift/transport/TBufferTransports.h>

#include <cub/cub.cuh>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

namespace experimental {

namespace parquet {

namespace utility {

template <typename U, typename T>
constexpr __host__ __device__ U roundUpDiv(U const num, T const chunk)
{
  return (num / chunk) + (num % chunk > 0);
}

template <typename U, typename T>
constexpr __host__ __device__ U roundUpTo(U const num, T const chunk)
{
  return roundUpDiv(num, chunk) * chunk;
}

} // namespace utility

struct PageInfo {
  PageInfo(uint8_t *page_data,
           std::size_t uncompressed_bytes,
           void* output_data_ptr,
           cudf::bitmask_type* output_null_mask_ptr,
           cudf::size_type output_null_mask_offset,
           cudf::size_type type_size,           
           uint8_t const* dict, 
           cudf::size_type num_rows,
           cudf::size_type src_col_index)
    : page_data(page_data),
      uncompressed_bytes(uncompressed_bytes),
      output_data_ptr(output_data_ptr),
      output_null_mask_ptr(output_null_mask_ptr),
      output_null_mask_offset(output_null_mask_offset),
      type_size(type_size),
      dict(dict),
      num_rows(num_rows),
      src_col_index(src_col_index)
  {
  }

  uint8_t const* page_data;
  std::size_t uncompressed_bytes;
  // Pointer to the data section in the output column of this page
  void* output_data_ptr;
  // Pointer to the null mask section in the output column of this page
  cudf::bitmask_type* output_null_mask_ptr;
  // Null mask of this page should be stored beginning *output_null_mask_offset* bits after
  // *output_null_mask_ptr*
  cudf::size_type output_null_mask_offset;
  cudf::size_type type_size;
  uint8_t const* dict;
  cudf::size_type num_rows;
  cudf::size_type src_col_index;
  // Buffer to hold the decompressed page
  // rmm::device_uvector<uint8_t> decompressed_buffer;
};

int64_t get_val(const uint8_t* ptr)
{
  int64_t rtv = 0;
  for (int i = 0; i < 8; i++) {
    rtv += (static_cast<int64_t>(ptr[i]) << (i * 8));
  }
  return rtv;
}

__device__ void copy_val1(uint8_t* dst, uint8_t const* src)
{  
  dst[0] = src[0];
}

__device__ void copy_val2(uint8_t* dst, uint8_t const* src)
{  
  uint16_t rtv = 0;
  for (int i = 0; i < 2; i++) {
    rtv += (static_cast<uint16_t>(src[i]) << (i * 8));
  }
  (reinterpret_cast<uint16_t*>(dst))[0] = rtv;
}

__device__ void copy_val4(uint8_t* dst, uint8_t const* src)
{  
  uint32_t rtv = 0;
  for (int i = 0; i < 4; i++) {  
    rtv += (static_cast<uint32_t>(src[i]) << (i * 8));
  }
  (reinterpret_cast<uint32_t*>(dst))[0] = rtv;
}

__device__ void copy_val8(uint8_t* dst, uint8_t const* src)
{  
  uint64_t rtv = 0;
  for (int i = 0; i < 8; i++) {
    rtv += (static_cast<uint64_t>(src[i]) << (i * 8));
  }
  (reinterpret_cast<uint64_t*>(dst))[0] = rtv;
}


/*
 * This helper function calculates the length of each bitpacked run or RLE run from the tag when
 * decoding the definition level.
 *
 * The length is encoded using ULEB128 (https://en.wikipedia.org/wiki/LEB128).
 */
__forceinline__ __device__ uint32_t calculate_run_length(const uint8_t*& current_ptr)
{
  uint64_t result = 0;
  uint64_t shift  = 0;
  while (true) {
    uint8_t current_val = *current_ptr;
    current_ptr++;
    result |= static_cast<uint64_t>(current_val & 0x7F) << shift;
    if (!(current_val & 0x80)) break;
    shift += 7;
  }
  return static_cast<uint32_t>(result >> 1);
}

/*
 * Assume the unset bits in *dst* are initialized to 0.
 */
template<int num_warps_per_block>
__device__ cudf::size_type copy_validity_bits_safe(cudf::bitmask_type* dst,
                          uint32_t dst_bit_offset,
                          const uint8_t* src,
                          uint32_t num_bytes,
                          int local_warp_id)
{
  const int warp_lane = threadIdx.x % 32;
  
  uint32_t end_bit_offset    = dst_bit_offset + (num_bytes * 8) - 1;
  uint32_t start_output_byte = dst_bit_offset / 8;
  uint32_t end_output_byte   = end_bit_offset / 8;
  uint32_t start_output_word = dst_bit_offset / 32;
  uint32_t end_output_word   = end_bit_offset / 32;
  uint32_t* output           = reinterpret_cast<uint32_t*>(dst);         
    
  typedef cub::WarpReduce<uint8_t> WarpReduce;
  __shared__ typename WarpReduce::TempStorage temp_storage[num_warps_per_block];
  cudf::size_type valid_bits = 0;

  // start from the first byte of the first word, which is not necessarily the first byte of the output.
  // each warp can process 32 bytes, or 8 bitmask words
  for (uint32_t i = start_output_word; i <= end_output_word; i += 8) {
    uint32_t output_byte = (i * 4) + warp_lane;
    
    uint8_t current_output_byte;
    if(output_byte < start_output_byte || output_byte > end_output_byte){
      current_output_byte = 0;
    } else if (output_byte * 8 < dst_bit_offset) {
      current_output_byte = src[0] << (dst_bit_offset - output_byte * 8);
    } else {
      uint32_t start_input_bit = output_byte * 8 - dst_bit_offset;
      uint32_t low_idx         = start_input_bit / 8;
      uint32_t offset          = start_input_bit - low_idx * 8;

      current_output_byte = src[low_idx] >> offset;
      if (offset && low_idx + 1 < num_bytes) current_output_byte |= (src[low_idx + 1] << (8 - offset));
    }
    
    // results will only be valid for lane 0. caller beware
    valid_bits += WarpReduce(temp_storage[local_warp_id]).Sum(__popc(current_output_byte));

    // shuffle bytes into groups of bitmask_type words so we can store via atomicOr.
    // there is a boundary condition where one block can be writing the trailing bits 
    // from one page which happen to overlap the leading bits in the same byte on a separate
    // page running on a different block.
    int word_index = output_byte / 4;
    
    auto const sub_word_index = warp_lane / 4; 
    auto const sub_word_lane = warp_lane % 4;
    uint32_t current_output = static_cast<uint32_t>(current_output_byte) << (sub_word_lane * 8);
    auto const mask = 0xf << sub_word_index;
    current_output |= __shfl_xor_sync(mask, current_output, 1, 4);
    current_output |= __shfl_xor_sync(mask, current_output, 2, 4);    
    if(warp_lane % 4 == 0 && word_index <= end_output_word){
      atomicOr(output + word_index, current_output);
    }    
  }

  // only valid for lane 0
  return valid_bits;
}

/*
 * Assume the unset bits in *dst* are initialized to 0.
 */
__device__ void copy_validity_bits(cudf::bitmask_type* dst,
                          uint32_t dst_bit_offset,
                          const uint8_t* src,
                          uint32_t num_bytes)
{
  const int warp_lane = threadIdx.x % 32;  

  uint8_t* output           = reinterpret_cast<uint8_t*>(dst);
  uint32_t end_bit_offset   = dst_bit_offset + (num_bytes * 8) - 1;
  uint32_t start_output_idx = dst_bit_offset / 8;
  uint32_t end_output_idx   = end_bit_offset / 8;

  for (uint32_t output_idx = start_output_idx + warp_lane; output_idx <= end_output_idx;
       output_idx += 32) {         
    uint8_t current_output = 0;
    if (output_idx * 8 < dst_bit_offset) {
      current_output = src[0] << (dst_bit_offset - output_idx * 8);
    } else {
      uint32_t start_input_bit = output_idx * 8 - dst_bit_offset;
      uint32_t low_idx         = start_input_bit / 8;
      uint32_t offset          = start_input_bit - low_idx * 8;

      current_output = src[low_idx] >> offset;
      if (offset && low_idx + 1 < num_bytes) current_output |= (src[low_idx + 1] << (8 - offset));
    }    

    // TODO: Is it safe? Is it possible for two thread to update at the same time?
    output[output_idx] |= current_output;
  }
}

/*
 * Assume the unset bits in *dst* are initialized to 0.
 */
__device__ void set_validity_bits_safe(cudf::bitmask_type* dst, uint32_t dst_bit_offset, uint32_t num_bits)
{
  const int warp_lane = threadIdx.x % 32;
  uint32_t start_idx  = dst_bit_offset / (sizeof(cudf::bitmask_type) * 8);
  uint32_t end_idx    = (dst_bit_offset + num_bits - 1) / (sizeof(cudf::bitmask_type) * 8);
      
  for (uint32_t output_idx = start_idx + warp_lane; output_idx <= end_idx; output_idx += 32) {
    cudf::bitmask_type all_one_mask = static_cast<cudf::bitmask_type>(-1);
    cudf::bitmask_type mask         = all_one_mask;

    uint32_t start_bit = output_idx * sizeof(cudf::bitmask_type) * 8;
    uint32_t end_bit   = start_bit + sizeof(cudf::bitmask_type) * 8;

    if (start_bit < dst_bit_offset) mask &= (all_one_mask << (dst_bit_offset - start_bit));
    if (end_bit > dst_bit_offset + num_bits){
      mask &= (all_one_mask >> (end_bit - dst_bit_offset - num_bits));
    }    
    atomicOr(dst + output_idx, mask);
  }
}


/*
 * Assume the unset bits in *dst* are initialized to 0.
 */
__device__ void set_validity_bits(cudf::bitmask_type* dst, uint32_t dst_bit_offset, uint32_t num_bits)
{
  const int warp_lane = threadIdx.x % 32;
  uint32_t start_idx  = dst_bit_offset / (sizeof(cudf::bitmask_type) * 8);
  uint32_t end_idx    = (dst_bit_offset + num_bits - 1) / (sizeof(cudf::bitmask_type) * 8);
  
  for (uint32_t output_idx = start_idx + warp_lane; output_idx <= end_idx; output_idx += 32) {
    cudf::bitmask_type all_one_mask = static_cast<cudf::bitmask_type>(-1);
    cudf::bitmask_type mask         = all_one_mask;

    uint32_t start_bit = output_idx * sizeof(cudf::bitmask_type) * 8;
    uint32_t end_bit   = start_bit + sizeof(cudf::bitmask_type) * 8;

    if (start_bit < dst_bit_offset) mask &= (all_one_mask << (dst_bit_offset - start_bit));
    if (end_bit > dst_bit_offset + num_bits)
      mask &= (all_one_mask >> (end_bit - dst_bit_offset - num_bits));

    dst[output_idx] |= mask;
  }
}

constexpr int dict_buf_size = 64;
struct dict_info {
  int           dict_val;
  uint32_t      dict_run;
  int           dict_bits;
  uint8_t const *data_start, *data_end;
  int           dict_pos;  
  uint32_t      dict_idx[dict_buf_size];
};

template <typename T>
inline __device__ T shuffle(T var, int lane = 0)
{
  return __shfl_sync(~0, var, lane);
}

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(const uint8_t*& cur, const uint8_t* end)
{
  uint32_t v = *cur++;
  if (v >= 0x80 && cur < end) {
    v = (v & 0x7f) | ((*cur++) << 7);
    if (v >= (0x80 << 7) && cur < end) {
      v = (v & ((0x7f << 7) | 0x7f)) | ((*cur++) << 14);
      if (v >= (0x80 << 14) && cur < end) {
        v = (v & ((0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 21);
        if (v >= (0x80 << 21) && cur < end) {
          v = (v & ((0x7f << 21) | (0x7f << 14) | (0x7f << 7) | 0x7f)) | ((*cur++) << 28);
        }
      }
    }
  }
  return v;
}

/**
 * @brief Performs RLE decoding of dictionary indexes
 *
 * @param[in,out] s Page state input/output
 * @param[in] target_pos Target index position in dict_idx buffer (may exceed this value by up to
 * 31)
 * @param[in] t Warp1 thread ID (0..31)
 *
 * @return The new output position
 */
__device__ int gpuDecodeDictionaryIndices(dict_info *s, int target_pos, int t)
{  
  const uint8_t* end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;      
      const uint8_t* cur = s->data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          int bytecnt = (dict_bits + 7) >> 3;
          if (cur + bytecnt <= end) {
            int32_t run_val = cur[0];
            if (bytecnt > 1) {
              run_val |= cur[1] << 8;
              if (bytecnt > 2) {
                run_val |= cur[2] << 16;
                if (bytecnt > 3) { run_val |= cur[3] << 24;
                }
              }
            }
            s->dict_val = run_val & ((1 << dict_bits) - 1);
          }
          cur += bytecnt;
        }
      }      
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len      = max(min(32, (int)(run >> 1) * 8), 1);
        batch_len_div8 = (batch_len + 7) >> 3;        
        run -= batch_len_div8 * 2;
        cur += batch_len_div8 * dict_bits;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->dict_run   = run;
      s->data_start = cur;
      is_literal    = run & 1;
      __threadfence_block();
    }
    __syncwarp();
    is_literal = shuffle(is_literal);
    batch_len  = shuffle(batch_len);
    if (t < batch_len) {
      int dict_idx = s->dict_val;
      if (is_literal) {
        int32_t ofs      = (t - ((batch_len + 7) & ~7)) * dict_bits;
        const uint8_t* p = s->data_start + (ofs >> 3);
        ofs &= 7;
        if (p < end) {
          uint32_t c = 8 - ofs;
          dict_idx   = (*p++) >> ofs;
          if (c < dict_bits && p < end) {
            dict_idx |= (*p++) << c;
            c += 8;
            if (c < dict_bits && p < end) {
              dict_idx |= (*p++) << c;
              c += 8;
              if (c < dict_bits && p < end) { dict_idx |= (*p++) << c; }
            }
          }
          dict_idx &= (1 << dict_bits) - 1;
        }
      }
      s->dict_idx[(pos + t) & (dict_buf_size - 1)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

__device__ void dict_buffer_to(dict_info *di, int target_pos, int t)
{  
  di->dict_pos = gpuDecodeDictionaryIndices(di, target_pos, t);
}

/**
 * CUDA kernel for decoding pages and copying the data and null masks to the output buffers.
 *
 * @param[in] page_data Array of length *num_pages*, where the `i`th entry is a pointer to the start
 * of `i`th page in uncompressed form.
 * @param[in] page_size Array of length *num_pages*, where the `i`th entry is the size of ith page.
 * @param[out] output_data Array of length *num_pages*, where the ith entry is a pointer to the data
 * buffer that will hold the decoded data of page `i`.
 * @param[out] output_bitmask Array of length *num_pages*, where the `i`th entry is a pointer to the
 * base of the null mask buffer. The start of the null mask for page `i` is *output_bitmask* plus
 * *output_bitmask_offset* bits.
 * @param[in] output_bitmask_offset Array of length *num_pages*, where the `i`th entry is the
 * distance in bits between the start of the null mask of page `i` to *output_bitmask*. Note the
 * unit is in bits not bytes.
 */
template <int num_warps_per_block>
__global__ void decode_pages_kernel(const void* const* page_data,
                                    const std::size_t* page_size,
                                    uint8_t* const* output_data,
                                    cudf::bitmask_type* const* output_bitmask,
                                    const cudf::size_type* output_bitmask_offset,
                                    const cudf::size_type* page_type_size,
                                    uint8_t const** dicts,
                                    cudf::size_type const* num_rows,
                                    cudf::size_type* output_null_counts,
                                    std::size_t num_pages)
{
  // using data_type = int64_t;

  const int warp_id       = blockIdx.x * num_warps_per_block + threadIdx.x / 32;
  const int local_warp_id = threadIdx.x / 32;  // warp id within a threadblock
  const int num_warps     = gridDim.x * num_warps_per_block;
  const int warp_lane     = threadIdx.x % 32;

  typedef cub::WarpScan<uint8_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[num_warps_per_block];  
  __shared__ dict_info block_dict_info[num_warps_per_block];

  // Assign each page to a warp
  for (std::size_t page_idx = warp_id; page_idx < num_pages; page_idx += num_warps) {    
    const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);    

    if(warp_lane == 0){      
      output_null_counts[page_idx] = 0;
    }

    // dictionary pages have nothing to decode.
    if(page_start == nullptr){
      continue;
    }     
    
    uint8_t const* page_data_cur         = page_start;    
    cudf::bitmask_type* page_bitmask_ptr = output_bitmask[page_idx];  // starts at the absolute beginning of column validity output

    // length of the encoded definition levels stored as 4 bytes little endian
    uint32_t definition_level_size;
    if(page_bitmask_ptr != nullptr){
      definition_level_size = page_data_cur[0] + (page_data_cur[1] << 8) + (page_data_cur[2] << 16) + (page_data_cur[3] << 24);
      page_data_cur += 4;
    } else {
      definition_level_size = 0;
    }
    
    // pointer to the definition levels currently being decoded
    const uint8_t* current_level_ptr = page_data_cur;     
    page_data_cur += definition_level_size; 
        
    // dictionary handling
    uint8_t const* dict = dicts[page_idx];
    dict_info* di = dict ? &block_dict_info[local_warp_id] : nullptr;
    if(di){
      if(warp_lane == 0){      
        di->dict_val = 0;
        di->dict_run = 0;
        di->dict_bits = *page_data_cur; 
        di->data_start = page_data_cur+1;
        di->data_end = page_start + page_size[page_idx];   
        di->dict_pos = 0; 
      }
      __syncwarp();
    }    

    // pointer to the start of the values section 
    const uint8_t* input_data_ptr          = page_data_cur;
    const uint8_t* current_data_ptr        = input_data_ptr;
    int current_data_pos                   = 0;                       // by value index. tracks current_data_ptr
    uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
    cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
    cudf::size_type type_size = page_type_size[page_idx];
    cudf::size_type values_processed = 0;
    cudf::size_type valid_count = 0;

    // Keep going until the definition levels have been completely parsed
    // Note that the end of the definition level section is the same as the start of the values
    // section
    while ((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)) ||
           (values_processed < num_rows[page_idx])) {
      
      uint8_t tag     = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
      uint32_t length = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);

      if (tag & 1) {
        // bitpacked run
        // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
        // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
        // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
        // null masks, we can directly copy from the bitpacked definition levels.
        std::size_t num_rounds = utility::roundUpDiv(length * 8, 32);
        for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
          std::size_t bit_idx         = round_idx * 32 + warp_lane;
          std::size_t byte_idx        = bit_idx / 8;
          std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

          uint8_t mask             = 0;
          uint8_t exclusive_output = 0; // output position per thread
          uint8_t warp_aggregate   = 0; // total # of values decoded for the whole warp

          if (byte_idx < length) {
            uint8_t current_byte = current_level_ptr[byte_idx];
            // mask will be either 0 or 1, indicating whether the current value is NULL
            mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
          }

          WarpScan(temp_storage[local_warp_id])
            .ExclusiveSum(mask, exclusive_output, warp_aggregate);
          
          if(dict){
            dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
          }

          if (mask) {
            auto const src = dict ? dict + (di->dict_idx[(current_data_pos + exclusive_output) & (dict_buf_size - 1)] * type_size)
                                  : current_data_ptr + exclusive_output * type_size;                                  

            switch(type_size){
            //case 1: copy_val1(current_output_ptr + (bit_idx * type_size), src); break;
            //case 2: copy_val2(current_output_ptr + (bit_idx * type_size), src); break;
            case 4: copy_val4(current_output_ptr + (bit_idx * type_size), src); break;
            case 8: copy_val8(current_output_ptr + (bit_idx * type_size), src); break;
            default: break;
            }
          }

          current_data_ptr += (warp_aggregate * type_size);
          current_data_pos += warp_aggregate;
        }
        
        if(page_bitmask_ptr != nullptr){
          valid_count += copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, current_bitmask_offset, current_level_ptr, length, local_warp_id);
          // copy_validity_bits(page_bitmask_ptr, current_bitmask_offset, current_level_ptr, length);
          current_level_ptr += length;
        }
                
        current_output_ptr += (length * 8 * type_size); 
        current_bitmask_offset += (length * 8);
        values_processed += (length * 8);
      } else {
        // RLE run
        // Again, for a flat data type, each value has 1 bit stored in the definition level: 0 or 1.
        // So the repeated value must be either 0 or 1. If the repeated value is 0, it means we have
        // repeated NULLs in the column. Since the NULL mask is initialized to 0, we do not need to
        // do anything. If the repeated value is 1, we need to copy the data, and set the null mask
        // to 1.
        //
        // For pages with no definition levels, pretend the validity value is just 1 and decode all
        // the values in 1 loop
        uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *current_level_ptr;

        if (repeated_value) {
          uint32_t output_base_idx = 0;
          do {
            int warp_aggregate = output_base_idx + 32 > length ? length - output_base_idx : 32;
            
            if(dict){
              dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
            }

            if(output_base_idx + warp_lane < length){              
              auto const output_idx = output_base_idx + warp_lane;
              auto const src = dict ? dict + (di->dict_idx[(current_data_pos + warp_lane) & (dict_buf_size - 1)] * type_size)
                                    : current_data_ptr + output_idx * type_size;                                    

              switch(type_size){
              //case 1: copy_val1(current_output_ptr + (output_idx * type_size), src); break;
              //case 2: copy_val2(current_output_ptr + (output_idx * type_size), src); break;
              case 4: copy_val4(current_output_ptr + (output_idx * type_size), src); break;
              case 8: copy_val8(current_output_ptr + (output_idx * type_size), src); break;
              default: break;
              }
            }

            current_data_pos += warp_aggregate;
            output_base_idx += warp_aggregate;            
          } while(output_base_idx < length);
                    
          current_data_ptr += (length * type_size);
          
          if(page_bitmask_ptr != nullptr){
            set_validity_bits_safe(page_bitmask_ptr, current_bitmask_offset, length);
            // set_validity_bits(page_bitmask_ptr, current_bitmask_offset, length);
          }
          valid_count += length;
        }
                
        if(page_bitmask_ptr != nullptr){
          current_level_ptr++;
        }
        current_output_ptr += (length * type_size);
        current_bitmask_offset += length;        
        values_processed += length;
      }
    }

    if(warp_lane == 0){
      // printf("WRITING (%lu): %d\n", page_idx, num_rows[page_idx] - valid_count);
      output_null_counts[page_idx] = num_rows[page_idx] - valid_count;
    }
  }  
}

std::pair<cudf::data_type, cudf::size_type> to_type(cudf::io::parquet::Type type, cudf::io::parquet::ConvertedType converted_type, int scale)
{
  // printf("T: %d, CT: %d\n", (int)type, (int)converted_type);
  switch(converted_type){
  case cudf::io::parquet::ConvertedType::DECIMAL:
    switch(type){
      case cudf::io::parquet::Type::INT64 : return {cudf::data_type{cudf::type_id::DECIMAL64, numeric::scale_type{-scale}}, 8};
      case cudf::io::parquet::Type::INT32 : return {cudf::data_type{cudf::type_id::DECIMAL32, numeric::scale_type{-scale}}, 4};

      default: return {cudf::data_type{cudf::type_id::EMPTY}, 0};
    }
    break;
  // these will need downconversion handling
  case cudf::io::parquet::ConvertedType::UINT_8: 
  case cudf::io::parquet::ConvertedType::INT_8:     
  case cudf::io::parquet::ConvertedType::UINT_16: 
  case cudf::io::parquet::ConvertedType::INT_16:
    return {cudf::data_type{cudf::type_id::EMPTY}, 0};

  // timestamp stuff needs special handling as well
  case cudf::io::parquet::ConvertedType::DATE: 
  case cudf::io::parquet::ConvertedType::TIME_MILLIS: 
  case cudf::io::parquet::ConvertedType::TIME_MICROS:     
  case cudf::io::parquet::ConvertedType::TIMESTAMP_MILLIS: 
  case cudf::io::parquet::ConvertedType::TIMESTAMP_MICROS:
    return {cudf::data_type{cudf::type_id::EMPTY}, 0};
  
  case cudf::io::parquet::ConvertedType::UINT_32: return {cudf::data_type{cudf::type_id::UINT32}, 4};
  case cudf::io::parquet::ConvertedType::UINT_64: return {cudf::data_type{cudf::type_id::UINT64}, 8};
  // case cudf::io::parquet::ConvertedType::UINT_64: return {cudf::data_type{cudf::type_id::EMPTY}, 0};
  default: break;
  }
  
  // other 
  switch(type){
  case cudf::io::parquet::Type::INT64 : return {cudf::data_type{cudf::type_id::INT64}, 8};
  case cudf::io::parquet::Type::INT32 : return {cudf::data_type{cudf::type_id::INT32}, 4};
  case cudf::io::parquet::Type::DOUBLE : return {cudf::data_type{cudf::type_id::FLOAT64}, 8};
  case cudf::io::parquet::Type::FLOAT : return {cudf::data_type{cudf::type_id::FLOAT32}, 4};
  default: break;
  }
  
  return {cudf::data_type{cudf::type_id::EMPTY}, 0};
}

std::optional<PageInfo> to_nvcomp(cudf::io::parquet::gpu::PageInfo const& page, cudf::io::parquet::gpu::ColumnChunkDesc const& chunk)
{   
  // reject lists.
  if(chunk.max_level[cudf::io::parquet::gpu::level_type::REPETITION] > 0){
    //printf("MAX REP: %d\n", chunk.max_level[cudf::io::parquet::gpu::level_type::REPETITION]);
    return std::nullopt;
  }
  if(chunk.max_level[cudf::io::parquet::gpu::level_type::DEFINITION] > 1){
    //printf("MAX DEF: %d\n", chunk.max_level[cudf::io::parquet::gpu::level_type::DEFINITION]);
    // printf("VMAP(%d): %lu\n", (int)chunk.max_level[cudf::io::parquet::gpu::level_type::DEFINITION], (int64_t)chunk.valid_map_simple);
    return std::nullopt;
  }

  auto const supported_encoding = page.encoding == cudf::io::parquet::Encoding::PLAIN || page.encoding == cudf::io::parquet::Encoding::PLAIN_DICTIONARY;
  auto const type = to_type(static_cast<cudf::io::parquet::Type>(chunk.data_type & 0x7), 
                            static_cast<cudf::io::parquet::ConvertedType>(chunk.converted_type), 
                            chunk.decimal_scale);  
  if(!supported_encoding || type.first.id() == cudf::type_id::EMPTY){
    /*
    if(!supported_encoding){
      printf("Unsupported page encoding : %d\n", static_cast<int>(page.encoding));
    }
    if(type.first.id() == cudf::type_id::EMPTY){
      printf("Unsupported cudf type : %d\n", static_cast<int>(type.first.id()));
    }
    */
    return std::nullopt;
  }

  //printf("Accepting: %d, %d\n", (int)chunk.data_type & 0x7, (int)chunk.converted_type);

  auto const row_index = page.chunk_row + chunk.start_row;
  return std::optional<PageInfo>({page.flags & cudf::io::parquet::gpu::PAGEINFO_FLAGS_DICTIONARY ? nullptr : page.page_data,
           static_cast<size_t>(page.uncompressed_page_size), 
           page.flags & cudf::io::parquet::gpu::PAGEINFO_FLAGS_DICTIONARY ? nullptr : static_cast<uint8_t*>(chunk.column_data_simple) + (row_index * type.second),
           page.flags & cudf::io::parquet::gpu::PAGEINFO_FLAGS_DICTIONARY ? nullptr : static_cast<cudf::bitmask_type*>(chunk.valid_map_simple),
           static_cast<cudf::size_type>(row_index),
           type.second,          
           page.encoding == cudf::io::parquet::Encoding::PLAIN_DICTIONARY ? chunk.dict_data : nullptr,
           page.num_input_values,
           chunk.src_col_index});
}

/**
 * Decode the decompressed pages.
 *
 * For each page, the decoder will read the `decompressed_buffer` in the `PageInfo` and copy the
 * decoded values and null masks to the output buffers specified in *PageInfo*.
 *
 * @param pages Pages to be decoded.
 */
std::pair<std::vector<cudf::size_type>, hostdevice_vector<cudf::size_type>> decode_values(std::vector<PageInfo> const& pages, rmm::cuda_stream_view stream)
{
  const auto num_pages = pages.size();

  printf("NVCOMP processing: %lu pages\n", pages.size());

  // This test is necessary because thrust::raw_pointer_cast(X.data()) could lead to error if the
  // size is 0.
  if (num_pages == 0){
     return {};
  }

  std::vector<void const*> decompressed_ptrs(num_pages);
  std::vector<std::size_t> uncompressed_bytes(num_pages);
  std::vector<uint8_t*> output_data_ptrs(num_pages);
  std::vector<cudf::bitmask_type*> output_null_mask_ptrs(num_pages);
  std::vector<cudf::size_type> output_null_mask_offsets(num_pages);
  std::vector<cudf::size_type> output_type_sizes(num_pages);
  std::vector<uint8_t const*> dicts(num_pages);
  std::vector<cudf::size_type> num_rows(num_pages);
  std::vector<cudf::size_type> output_src_col_indices(num_pages);

  for (std::size_t page_idx = 0; page_idx < num_pages; page_idx++) {
    decompressed_ptrs[page_idx]        = pages[page_idx].page_data;   // will be null for dictionary pages
    uncompressed_bytes[page_idx]       = pages[page_idx].uncompressed_bytes;
    output_data_ptrs[page_idx]         = static_cast<uint8_t*>(pages[page_idx].output_data_ptr);
    output_null_mask_ptrs[page_idx]    = pages[page_idx].output_null_mask_ptr;
    output_null_mask_offsets[page_idx] = pages[page_idx].output_null_mask_offset;
    output_type_sizes[page_idx] = pages[page_idx].type_size;
    dicts[page_idx] = pages[page_idx].dict;
    num_rows[page_idx] = pages[page_idx].num_rows;
    output_src_col_indices[page_idx] = pages[page_idx].src_col_index;
  }

  rmm::device_vector<void const*> device_decompressed_ptrs             = decompressed_ptrs;
  rmm::device_vector<std::size_t> device_uncompressed_bytes            = uncompressed_bytes;
  rmm::device_vector<uint8_t*> device_output_data_ptrs                 = output_data_ptrs;
  rmm::device_vector<cudf::bitmask_type*> device_output_null_mask_ptrs = output_null_mask_ptrs;
  rmm::device_vector<cudf::size_type> device_output_null_mask_offsets  = output_null_mask_offsets;
  rmm::device_vector<cudf::size_type> device_output_type_sizes  = output_type_sizes;
  rmm::device_vector<uint8_t const*> device_dicts = dicts;
  rmm::device_vector<cudf::size_type> device_num_rows = num_rows;  
  hostdevice_vector<cudf::size_type> device_output_null_counts(num_pages, stream);

  constexpr int num_warps_per_block = 4;

  decode_pages_kernel<num_warps_per_block>
    <<<utility::roundUpDiv(num_pages, 4), num_warps_per_block * 32, 0, stream.value()>>>(
      thrust::raw_pointer_cast(device_decompressed_ptrs.data()),
      thrust::raw_pointer_cast(device_uncompressed_bytes.data()),
      thrust::raw_pointer_cast(device_output_data_ptrs.data()),
      thrust::raw_pointer_cast(device_output_null_mask_ptrs.data()),
      thrust::raw_pointer_cast(device_output_null_mask_offsets.data()),
      thrust::raw_pointer_cast(device_output_type_sizes.data()),
      thrust::raw_pointer_cast(device_dicts.data()),
      thrust::raw_pointer_cast(device_num_rows.data()),
      device_output_null_counts.device_ptr(),
      num_pages);

  return {std::move(output_src_col_indices), std::move(device_output_null_counts)};
}

// given an input list of cuIO parquet reader pages, dispatch as many of them as we can using the nvcomp
// reader. return a vector of pages that could not be processed.
std::tuple<hostdevice_vector<cudf::io::parquet::gpu::PageInfo>,
           std::vector<cudf::size_type>,
           hostdevice_vector<cudf::size_type>>                                  
decode_relevant_pages(hostdevice_vector<cudf::io::parquet::gpu::ColumnChunkDesc>& chunks,
                      hostdevice_vector<cudf::io::parquet::gpu::PageInfo>& _pages,
                      rmm::cuda_stream_view stream)
{  
  hostdevice_vector<cudf::io::parquet::gpu::PageInfo> remainder(0, _pages.size(), stream);
  std::vector<PageInfo> nvcomp_pages;
  nvcomp_pages.reserve(_pages.size());
  std::for_each(_pages.begin(), _pages.end(), [&](cudf::io::parquet::gpu::PageInfo const &p){
    auto nvcomp = to_nvcomp(p, chunks[p.chunk_idx]);
    if(nvcomp.has_value()){
      nvcomp_pages.push_back(nvcomp.value());
    } else {
      remainder.insert(p);
    }
  });

  // invoke nvcomp decoding
  auto [src_col_indices, null_counts] = decode_values(nvcomp_pages, stream);
  
  // return the remainder of the pages to cuIO for decoding.
  remainder.host_to_device(stream);
  return {std::move(remainder), std::move(src_col_indices), std::move(null_counts)};
}

} // namespace parquet

} // namespace experimental