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
// #include "db_test_timer.hpp"

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/device_vector.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <io/utilities/hostdevice_vector.hpp>

#include <cub/cub.cuh>

#include <thrust/binary_search.h>

#include <cooperative_groups.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

bool Use_nvcomp_decode_path2 = true;

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
           cudf::size_type src_col_index,
           int dict_page_index)
    : page_data(page_data),
      uncompressed_bytes(uncompressed_bytes),
      output_data_ptr(output_data_ptr),
      output_null_mask_ptr(output_null_mask_ptr),
      output_null_mask_offset(output_null_mask_offset),
      type_size(type_size),
      dict(dict),
      num_rows(num_rows),
      src_col_index(src_col_index),
      dict_page_index(dict_page_index)
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
  int dict_page_index;
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
  // memcpy(dst, src, 2);  
  uint16_t rtv = 0;
  for (int i = 0; i < 2; i++) {
    rtv += (static_cast<uint16_t>(src[i]) << (i * 8));
  }
  (reinterpret_cast<uint16_t*>(dst))[0] = rtv;  
}

__device__ void copy_val4(uint8_t* dst, uint8_t const* src)
{  
  // (reinterpret_cast<uint32_t*>(dst))[0] = (reinterpret_cast<uint32_t*>(((uint64_t)src) & ~0x3)[0]);
  
  // memcpy(dst, src, 4);
    
  uint32_t rtv = 0;
  for (int i = 0; i < 4; i++) {  
    rtv += (static_cast<uint32_t>(src[i]) << (i * 8));
  }
  (reinterpret_cast<uint32_t*>(dst))[0] = rtv;
}

__device__ void copy_val8(uint8_t* dst, uint8_t const* src)
{  
  // memcpy(dst, src, 8);  
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

__forceinline__ __device__ uint32_t calculate_run_length2(const uint8_t* current_ptr)
{
  uint64_t result = 0;
  uint64_t shift  = 0;
  int ptr_offset = 0;
  while (true) {
    uint8_t current_val = *(current_ptr + ptr_offset);
    ptr_offset++;
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
                                                   uint32_t bit_offset,      // offset with in the page
                                                   uint32_t base_bit_offset, // base offset of ht epage
                                                   int num_rows,             // num rows in the page
                                                   const uint8_t* src,
                                                   uint32_t num_bytes,
                                                   int local_warp_id
                                                   /*,int total_rows = 0*/)
{
  int const warp_lane = threadIdx.x % 32;

  uint32_t const dst_bit_offset = bit_offset + base_bit_offset;
  // we may have 1 byte (8 rows) of info, but only say 2 of the bits are relevant because we are at the last
  // 2 rows.  we need to cap the # of bits we're writing so that we don't go past the end.
  uint32_t const num_bits = min((num_bytes * 8), num_rows - bit_offset);
  
  uint32_t const end_bit_offset    = dst_bit_offset + num_bits - 1;
  uint32_t const start_output_byte = dst_bit_offset / 8;
  uint32_t const end_output_byte   = end_bit_offset / 8;
  uint32_t const start_output_word = dst_bit_offset / 32;
  uint32_t const end_output_word   = end_bit_offset / 32;
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

// (kernel) 4
// constexpr int dict_buf_count = 32;
// constexpr int dict_buf_size = dict_buf_count * 4;   // must be a multiple of 2

// kernel 5
constexpr int dict_buf_count = 128;
constexpr int dict_buf_size = dict_buf_count * 4;   // must be a multiple of 2
struct dict_info {
  int dict_val;
  int dict_run;
  int dict_bits;
  int dict_pos;     // logical position.
  int dict_batch_len;

  // in the process of decoding, we may end up reading ahead of
  // our logical position.  this tracks where we can currently
  // decode up to.
  int read_pos;  
  
  uint8_t const *data_start, *data_end;   
  uint32_t      dict_idx[dict_buf_size];

  __device__ void copy_pos(dict_info const& d)
  {
    dict_val = d.dict_val;
    dict_run = d.dict_run;
    dict_bits = d.dict_bits;
    dict_pos = d.dict_pos;
    dict_batch_len = d.dict_batch_len;
    
    read_pos = d.read_pos;

    // TODO: redundant if we initialized up front
    data_start = d.data_start;
    data_end = d.data_end;
  }
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
__device__ int gpuDecodeDictionaryIndices2(dict_info *s, int target_pos, int t)
{  
  const uint8_t* end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    // determine run length and type
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

    // decode the run.
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

__device__ int gpuDictionaryDecodeTo(dict_info *s, int target_pos, int t, int page_idx, int local_warp_id)
{
  // should never be attempting to decode further than we have read up to
  if(!t && target_pos > s->read_pos){
    printf("gpuDictionaryDecodeTo error!\n");
  }

  /*
  if(page_idx == 1 && t == 0 && target_pos < 256){
    printf("DT(%d): target_pos(%d), dict_pos(%d), read_pos(%d)\n", local_warp_id, target_pos, s->dict_pos, s->read_pos);
  }
  */

  bool const is_literal = s->dict_run & 1;
  auto const batch_len  = target_pos - s->dict_pos;
  if (t < batch_len) {
    int dict_idx = s->dict_val;
    if (is_literal) {      
      int32_t ofs      = ((t + (s->dict_batch_len - (s->read_pos - s->dict_pos))) - ((s->dict_batch_len + 7) & ~7)) * s->dict_bits;
      const uint8_t* p = s->data_start + (ofs >> 3);      
      ofs &= 7;
      /*
      if(page_idx == 1 && target_pos < 256 && t == 0){
        printf("I0: dict_val(%d), dict_batch_len(%d), ofs(%d)\n", s->dict_val, s->dict_batch_len, ofs);
      }
      */
      if (p < s->data_end) {
        uint32_t c = 8 - ofs;
        dict_idx   = (*p++) >> ofs;
        if (c < s->dict_bits && p < s->data_end) {
          dict_idx |= (*p++) << c;
          c += 8;
          if (c < s->dict_bits && p < s->data_end) {
            dict_idx |= (*p++) << c;
            c += 8;
            if (c < s->dict_bits && p < s->data_end) { dict_idx |= (*p++) << c; }
          }
        }
        dict_idx &= (1 << s->dict_bits) - 1;
      }
    }

    /*
    if(page_idx == 1 && target_pos < 256 && t == 0){
      printf("I: dict_val(%d), pos(%d), pos+t(%d), dict_idx(%d)\n", s->dict_val, s->dict_pos, s->dict_pos+t, dict_idx);
    }
    */

    auto const pos = s->dict_pos + t;
    s->dict_idx[pos & (dict_buf_size - 1)] = dict_idx;        
    /*
    if(target_pos < 256 && page_idx == 1 && local_warp_id == 1){
      printf("V(%d): pos(%d), dict_idx(%d)\n", local_warp_id, pos, dict_idx); 
    }
    */
  }
  if(!t){
    s->dict_pos += batch_len;
  } 
  /* 
  if(target_pos < 256 && page_idx == 1 && local_warp_id <= 1){
    printf("VEXIT (%d)\n", local_warp_id);
  }
  */
  __syncwarp();
  return s->dict_pos;
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
__device__ void gpuDecodeDictionaryIndices(dict_info *s, int target_pos, int t, int page_idx, int local_warp_id)
{
  const uint8_t* end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;  
  
  do {    
    /*
    if(local_warp_id == 1 && t == 0 && target_pos < 256 && page_idx == 1){
      printf("E(%d): target_pos(%d), dict_pos(%d), read_pos(%d), dict_val(%d)\n", local_warp_id, target_pos, s->dict_pos, s->read_pos, s->dict_val);
    } 
    */   
    
    // decode what we can without advancing further into the stream
    pos = gpuDictionaryDecodeTo(s, min(target_pos, s->read_pos), t, page_idx, local_warp_id);    
    /*
    if(local_warp_id == 1 && t == 0 && target_pos < 256 && page_idx == 1){
      printf("E2(%d): target_pos(%d), dict_pos(%d), read_pos(%d), dict_val(%d)\n", local_warp_id, target_pos, s->dict_pos, s->read_pos, s->dict_val);
    } 
    */
    // decode what we can without advancing further into the stream
    // done.
    if(pos == target_pos){
      break;
    }
    if(s->read_pos != s->dict_pos){
      printf("gpuDecodeDictionaryIndices error 0\n");
      return;
    }

    // otherwise, read the next run.    
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
      int batch_len;
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
      // __threadfence_block();
      s->dict_batch_len = batch_len;
      s->read_pos += batch_len;
    }
    __syncwarp();    
  } while(1);

  /*
  if(page_idx == 1 && t == 0 && target_pos < 256 && local_warp_id == 1){
    printf("EXIT(%d), dict_val(%d)\n", local_warp_id, s->dict_val);
  }
  */
}

__device__ void gpuAdvanceDictionaryIndices(dict_info *s, int target_pos, int page_idx)
{
  const uint8_t* end = s->data_end;
  int dict_bits      = s->dict_bits;    

  do {        
    /*
    if(page_idx == 1 && target_pos < 256){
      printf("ADV0: target_pos(%d), dict_pos(%d), read_pos(%d)\n", target_pos, s->dict_pos, s->read_pos);
    } 
    */   

    // advance as far as we can.
    s->dict_pos = min(target_pos, s->read_pos);        
    /*
    if(page_idx == 1 && target_pos < 256){
      printf("ADV1: target_pos(%d), dict_pos(%d), read_pos(%d)\n", target_pos, s->dict_pos, s->read_pos);
    }
    */
    
    // done.
    if(s->dict_pos == target_pos){
      break;
    }
    if(s->read_pos != s->dict_pos){
      printf("gpuAdvanceDictionaryIndices error 0\n");
      return;
    }
    
    //printf("B(%d)\n", page_idx);    

    // otherwise, read the next run.
    int batch_len;    
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
    
    // we can now decode this much further.    
    s->read_pos += batch_len;
  } while(1);

  //printf("C(%d)\n", page_idx);        

  //__syncwarp();  
  /*
  if(page_idx == 1 && target_pos < 256){
    printf("ADVEXIT\n");
  }
  */
}

__device__ void dict_buffer_to2(dict_info *di, int target_pos, int t)
{  
  di->dict_pos = gpuDecodeDictionaryIndices2(di, target_pos, t);
}

__device__ void dict_buffer_to(dict_info *di, int target_pos, int t, int page_idx, int local_warp_id)
{  
  gpuDecodeDictionaryIndices(di, target_pos, t, page_idx, local_warp_id);
}

__device__ void dict_advance_to(dict_info *di, int target_pos, int page_idx)
{  
  gpuAdvanceDictionaryIndices(di, target_pos, page_idx);
}

struct work_unit {
  uint8_t         tag;
  uint8_t const   *level_ptr;
  uint8_t         *output_ptr;     
  int             length;
  int             bitmask_offset;  
  int             input_pos;
  int             input_length;
  
  dict_info       dict;  
};

struct rle3_run {
  uint8_t         tag;
  int             length;
  uint8_t const*  data;
};

struct rle3_stream {
  uint8_t const* cur;
  uint8_t const* end;
};

__forceinline__ __device__ thrust::tuple<int, int, uint8_t> calculate_run_length_and_tag(const uint8_t* cur, const uint8_t* end)
{
  uint64_t result = 0;
  uint64_t shift  = 0;  
  uint8_t tag = *cur;
  int offset = 0;
  while (true) {
    if(cur + offset >= end){
      break;
    } 
    uint8_t current_val = *(cur + offset);
    offset++;
    result |= static_cast<uint64_t>(current_val & 0x7F) << shift;
    if(result > 2000000000){
      break;
    }
    if (!(current_val & 0x80)) break;
    shift += 7;
  }

  auto const length = static_cast<uint32_t>(result >> 1);       // # of values
  return {length, offset, tag};
}

#if 0
// An attempt to speculatively decode an inherently serial stream of data in parallel.
//
// definition levels, repetition levels and dictionary indices are encoded in this format.
// if the return value is < requested_runs, it means we're at the end of the stream.
template<int num_warps_per_block>
__device__ int generate_rle3_runs(rle3_run *runs, int requested_runs, rle3_stream &stream)
{  
  int const warp_id = threadIdx.x / 32;
  int const warp_lane = threadIdx.x % 32;
  int const t = (warp_id) * 32 + warp_lane; 

  /*
  constexpr int max_runs = num_warps_per_block * 32;
  __shared__ int run_starts[max_runs];
  __shared__ int num_run_candidates;
  */
  
  /*
  if(!t){
    run_starts[0] = 0;
  }
  */

  auto stream_start = stream.cur;
  
  int num_runs = 0;
  while(num_runs < requested_runs){        
    #if 0
    // identify candidates in the next 32 bytes using warp 0.
    // TODO: do it across the whole block.
    if(warp_id == 0){
      auto const byte_index = warp_lane+1;

      uint8_t const byte = stream.cur + byte_index < stream.end ? stream.cur[byte_index] : 0;      
      // identify terminator candidates and all possible start points
      bool const candidate = stream.cur + byte_index < stream.end ? (!(byte & 0x80)) : false;
      if((stream.cur + byte_index) - stream_start == 45){
        printf("45 (byte_index %d): (%d) %d (%d) (%d, %d)\n", byte_index, stream.cur[byte_index-1], byte, stream.cur[byte_index+1], (!(byte & 0x80)), stream.cur + byte_index < stream.end);
      }
      int const candidates = __ballot_sync(0xffffffff, candidate);
      int const num_candidates = __popc(candidates);
      if(warp_lane == 0){
        num_run_candidates = num_candidates + 1;
      }
      if(candidate){
        auto const start_index = __popc(candidates & ((1 << warp_lane) - 1)) + 1;
        if(start_index < max_runs){
          run_starts[start_index] = byte_index;

          printf("L(%d): run_starts[%d] = %d (%d)\n", warp_lane, start_index, run_starts[start_index], (int)byte);
        }
      }
    }
    __syncthreads();
    #endif

    constexpr num_threads = num_warps_per_block * 32;

    __shared__ run_lengths[num_threads];
    __shared__ run_level_lengths[num_threads];
    __shared__ run_tags[num_threads];
    if(stream.cur + t < (stream.cur + stream.end){
      thrust::tie(run_lengths[warp_id], run_level_lengths[warp_id], run_tags[warp_id]) = calculate_run_length_and_tag(stream.cur + t, stream.end);
    }

    // decode all the possible run lengths and tags. not all of these results will be valid. but we
    // can march from the very first run, which we know -is- valid and pick the valid ones.
    // use a seperate warp to do each one to avoid thread divergence within one warp
    __shared__ int run_lengths[max_runs];         // length in values
    __shared__ int run_level_lengths[max_runs];   // length in bytes of the level data
    __shared__ uint8_t run_tags[max_runs];
    int max_runs = min(num_
    if(warp_id < num_run_candidates && warp_lane == 0){
      thrust::tie(run_lengths[warp_id], run_level_lengths[warp_id], run_tags[warp_id]) = calculate_run_length_and_tag(stream.cur + run_starts[warp_id], stream.end);
      // uint64_t dist = (stream.cur + run_starts[warp_id]) - stream_start;
      // printf("CRLT(%d): dist(%lu), %d, %d, %d, %d\n", warp_id, dist, run_starts[warp_id], run_lengths[warp_id], run_level_lengths[warp_id], run_tags[warp_id]);
    }
    __syncthreads();

    int 
    while(nu

    /*
    __syncthreads();
    if(t == 0){
      for(int idx=0; idx<num_run_candidates; idx++){
        printf("C(%d): length(%d), level_length(%d), tag(%d, %d)\n", idx, run_lengths[idx], run_level_lengths[idx], (int)run_tags[idx], (int)(run_tags[idx] & 1 ? 1 : 0));
      }
    }
    __syncthreads();    
    */

    // now march them to figure out which ones are valid, emitting runs along the way      
    if(!t){
      int cur_start = 0;
      while(cur_start < num_run_candidates){        
        runs[num_runs].tag = run_tags[cur_start];
        runs[num_runs].length = run_lengths[cur_start];        
        runs[num_runs].data = stream.cur;
                                
        int next_start_offset = run_starts[cur_start] + run_level_lengths[cur_start];

        uint64_t dist = stream.cur - stream_start;
        printf("START(%d, %d, %lu): %d, %d, %d, nso(%d)\n", num_runs, cur_start, dist, run_lengths[cur_start], run_tags[cur_start], run_level_lengths[cur_start], next_start_offset);

        stream.cur += run_level_lengths[cur_start];
        
        cur_start = thrust::lower_bound(thrust::seq, run_starts + cur_start + 1, run_starts + num_run_candidates, next_start_offset) - run_starts;        

        num_runs++;
      }
    }   
    __syncthreads();
  }

  if(!t){
    printf("DONE\n");
  }
  __syncthreads();  

  return num_runs;
}
#endif

// An attempt to speculatively decode an inherently serial stream of data in parallel.
//
// definition levels, repetition levels and dictionary indices are encoded in this format.
// if the return value is < requested_runs, it means we're at the end of the stream.
template<int num_warps_per_block>
__device__ int generate_rle3_runs(rle3_run *runs, int requested_runs, rle3_stream &stream)
{  
  int const warp_id = threadIdx.x / 32;
  int const warp_lane = threadIdx.x % 32;
  int const t = (warp_id * 32) + warp_lane;

  constexpr int num_threads = num_warps_per_block * 32;
  __shared__ int run_lengths[num_threads];
  __shared__ int run_level_lengths[num_threads];
  __shared__ uint8_t run_tags[num_threads];

  auto stream_start = stream.cur;  
  
  __shared__ int num_runs;
  num_runs = 0;  
  while(num_runs < requested_runs){    
    // pretend every possible byte starts a run. get lengths, level lengths and tags.
    if(stream.cur + t < stream.end){
      thrust::tie(run_lengths[t], run_level_lengths[t], run_tags[t]) = calculate_run_length_and_tag(stream.cur + t, stream.end);      
    }
    __syncthreads();

    // now march them to figure out which ones are valid, emitting runs along the way        
    if(!t){
      // index 0 is always a real start                  
      int cur_start = 0;
      while(num_runs < requested_runs && cur_start < num_threads && stream.cur < stream.end){
        runs[num_runs].tag = run_tags[cur_start];
        runs[num_runs].length = run_lengths[cur_start];        
        runs[num_runs].data = stream.cur + run_level_lengths[cur_start];        

                              // the header                  // packed bits after the header                             
        auto const consumed = run_level_lengths[cur_start] + ((run_tags[cur_start] & 1) ? run_lengths[cur_start] : 1);

        stream.cur += consumed;
        cur_start += consumed;

        num_runs++;
      }       
    }           
    __syncthreads();

    if(stream.cur >= stream.end){
      break;
    }
  }

  __syncthreads();
  return num_runs;
}

constexpr int k5_loop_value_count = dict_buf_count;  // 32

__forceinline__ __device__
void compute_round_info(int warp_lane,
                        int num_values, 
                        int round_mask[2][2],
                        int round_count[2][2],
                        int round_output_index[2][32],
                        uint8_t const* current_level_ptr, 
                        int buf)
{
  typedef cub::WarpScan<uint8_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage;  

  // will never be more than 2
  std::size_t num_rounds = utility::roundUpDiv(num_values, 32);
  for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) { 
    std::size_t bit_idx         = round_idx * 32 + warp_lane;
    std::size_t byte_idx        = bit_idx / 8;
    std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

    uint8_t valid            = 0;
    uint8_t exclusive_output = 0; // output position per thread
    uint8_t warp_aggregate   = 0; // total # of values decoded for the whole warp

    if (byte_idx < (num_values / 8)) {
      uint8_t current_byte = current_level_ptr[byte_idx];
      // mask will be either 0 or 1, indicating whether the current value is NULL
      valid = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
    }
    
    WarpScan(temp_storage).ExclusiveSum(valid, exclusive_output, warp_aggregate);

    auto const _round_mask = __ballot_sync(0xffffffff, valid);;
    if(warp_lane == 0){
      round_mask[buf][0] =_round_mask;
      round_count[buf][0] = warp_aggregate;
    }
    round_output_index[buf][warp_lane] = exclusive_output;
  }
}

template <int num_warps_per_block>
__global__ void decode_pages_kernel6(const void* const* page_data,
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
  const int warp_id       = threadIdx.x / 32;  // warp id within a threadblock
  const int page_idx      = blockIdx.x;
  const int warp_lane     = threadIdx.x % 32;  

  const bool anchor_thread = warp_id == 0 && warp_lane == 0;  

  // page setup
  const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);
  // have to set this to 0 before we exit.
  if(anchor_thread){      
    output_null_counts[page_idx] = 0;    
  }

  // if we're out of range or if this has no data (dictionary pages will have a nullptr here)
  if(page_idx >= num_pages || page_data[page_idx] == nullptr){
    return;
  }  
  
  // all shared data
  typedef cub::WarpScan<uint8_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage;  
  __shared__ dict_info di;  

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
  uint8_t const* current_level_ptr = page_data_cur;
  auto level_start = current_level_ptr;
  page_data_cur += definition_level_size; 

  // dictionary handling
  uint8_t const* dict = dicts[page_idx];
  if(dict && anchor_thread){    
    di.dict_val = 0;
    di.dict_run = 0;
    di.dict_bits = *page_data_cur; 
    di.data_start = page_data_cur+1;
    di.data_end = page_start + page_size[page_idx];   
    di.dict_pos = 0;     
    di.read_pos = 0;
    di.dict_batch_len = 0;
  }

  // pointer to the start of the values section 
  const uint8_t* input_data_ptr          = page_data_cur;
  __shared__ int current_input_pos;
  if(anchor_thread){
    current_input_pos = 0;
  }
  uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
  cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
  cudf::size_type type_size = page_type_size[page_idx];
  int valid_count = 0;  

  __syncthreads();
  
  // loop {
  //   decode level data (should this be warpy?)
  //   loop { if no dictionary data, all values are processed on the first iteration
  //     warp 0 decodes dictionary data if necessary
  //     warp 1 decodes values
  //   }
  //   warp 2 decode validity
  // }    

  // Keep going until the definition levels have been completely parsed
  // Note that the end of the definition level section is the same as the start of the values
  // section
  int values_processed = 0;
  while (reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)){
    // decode header stuff. should only 1 warp do this? I dunno.
    uint8_t  const tag         = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
    uint32_t const run_length  = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);
    // if this is a bitpacked run, there are 8 values for each byte in the level stream
    auto const     run_num_values = (tag & 1) ? run_length * 8 : run_length;

    // if we've got a dictionary, we have to process values in dict_buf_count chunks. otherwise
    // we can decode it all in one shot  
    // IMPORTANT: dict_buf_count must be a multiple of 8.      
    int num_values = dict ? min(dict_buf_count, run_num_values) : run_num_values;

    // warp 0 decodes the dictionary data (for the first iteration of the loop)
    if(dict){
      if(warp_id == 0){
        // note that we are only dictionary indices for non-null inputs.  So even
        // though the number of values in the level data (run_num_values) may be N, the number of dictionary
        // indices we read may be < N.
        dict_buffer_to2(&di, current_input_pos + (dict_buf_count * 2), warp_lane);      
      }
      __syncthreads();
    }

    int run_values_processed = 0;
    while(run_values_processed < run_num_values){      
      // warp 0 decodes dictionary data (for the next iteration of the loop)
      if(dict && warp_id == 0){
        // note that we are only dictionary indices for non-null inputs.  So even
        // though the number of values in the level data (run_num_values) may be N, the number of dictionary
        // indices we read may be < N.
        dict_buffer_to2(&di, current_input_pos + (dict_buf_count * 2), warp_lane);        
      }

      // warp 1 decodes values
      if(warp_id == 1){
        int _current_input_pos = current_input_pos;
        auto _current_input_ptr = input_data_ptr + (_current_input_pos * type_size);

        // bitpacked run
        // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
        // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
        // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
        // null masks, we can directly copy from the bitpacked definition levels.
        if(tag & 1){          
          std::size_t num_rounds = utility::roundUpDiv(num_values, 32);
          for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
            std::size_t bit_idx         = round_idx * 32 + warp_lane;
            std::size_t byte_idx        = bit_idx / 8;
            std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

            uint8_t mask             = 0;
            uint8_t exclusive_output = 0; // output position per thread
            uint8_t warp_aggregate   = 0; // total # of values decoded for the whole warp

            if (byte_idx < (num_values / 8)) {
              uint8_t current_byte = current_level_ptr[byte_idx];
              // mask will be either 0 or 1, indicating whether the current value is NULL
              mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
            }

            WarpScan(temp_storage).ExclusiveSum(mask, exclusive_output, warp_aggregate);
            
            if (mask) {
              auto const dict_pos = dict ? (_current_input_pos + exclusive_output) & (dict_buf_size - 1) : 0;
              auto const dict_val = dict ? di.dict_idx[dict_pos] : 0;
              auto const src = dict ? dict + (dict_val * type_size)
                                    : _current_input_ptr + exclusive_output * type_size;

              /*
              uint64_t output_pos = ((current_output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;              
              if(page_idx == 1 && output_pos < 256){                
                printf("COPYB0(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
              } 
              */              
              switch(type_size){
              case 4: copy_val4(current_output_ptr + (bit_idx * type_size), src); break;
              case 8: copy_val8(current_output_ptr + (bit_idx * type_size), src); break;
              default: break;
              }
            }

            _current_input_ptr += (warp_aggregate * type_size);
            _current_input_pos += warp_aggregate;
          }
        } 
        // RLE run
        // Again, for a flat data type, each value has 1 bit stored in the definition level: 0 or 1.
        // So the repeated value must be either 0 or 1. If the repeated value is 0, it means we have
        // repeated NULLs in the column. Since the NULL mask is initialized to 0, we do not need to
        // do anything. If the repeated value is 1, we need to copy the data, and set the null mask
        // to 1.
        //
        // For pages with no definition levels, pretend the validity value is just 1 and decode all
        // the values in 1 loop          
        else {
          uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *current_level_ptr;
          if (repeated_value) {
            uint32_t output_base_idx = 0;
            do {
              int warp_aggregate = output_base_idx + 32 > num_values ? num_values - output_base_idx : 32;
              
              if(output_base_idx + warp_lane < num_values){
                auto const output_idx = output_base_idx + warp_lane;

                auto const dict_pos = dict ? (_current_input_pos + warp_lane) & (dict_buf_size - 1) : 0;
                auto const dict_val = dict ? di.dict_idx[dict_pos] : 0;
                auto const src = dict ? dict + (dict_val * type_size)
                                      : _current_input_ptr + output_idx * type_size;
                
                /*
                uint64_t output_pos = ((current_output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;                
                if(page_idx == 1 && output_pos < 256){                  
                  printf("COPYB1(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
                }
                */

                switch(type_size){
                case 4: copy_val4(current_output_ptr + (output_idx * type_size), src); break;
                case 8: copy_val8(current_output_ptr + (output_idx * type_size), src); break;
                default: break;
                }
              }

              _current_input_pos += warp_aggregate;
              output_base_idx += warp_aggregate;
            } while(output_base_idx < num_values);
          }
        }

        // warp 0 needs to know how many actual non-null values we processed so it 
        // can buffer the dictionary appropriately
        if(warp_lane == 0){
          current_input_pos = _current_input_pos;
        }  
      }
      // warp 2 decodes validity
      else if(warp_id == 2){
        // bitpacked
        if(tag & 1){
          if(page_bitmask_ptr != nullptr){            
            valid_count += copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, current_bitmask_offset, current_level_ptr, num_values/8, warp_id);
          } else {
            printf("THIS SHOULDNT HAPPEN\n");
          }
        }
        // repeated
        else {
          // if this is not a nullable column or if the repeated value is 1 all the values are valid
          bool const valid = page_bitmask_ptr ? *current_level_ptr : true;
          if(valid){
            if(page_bitmask_ptr){
              set_validity_bits_safe(page_bitmask_ptr, current_bitmask_offset, num_values);
            }
            valid_count += num_values;
          }
        }
      }
      
      // everyone increments      
      current_output_ptr += (num_values * type_size); 
      current_bitmask_offset += num_values;
      if((tag & 1) && page_bitmask_ptr){        
        // 1 bit per value, so 8 values per byte.
        current_level_ptr += num_values / 8;
      } 
      run_values_processed += num_values;

      // next batch of values
      num_values = dict ? min(dict_buf_count, run_num_values - run_values_processed) : run_num_values;
     
      __syncthreads();
    } // inner value decoding loop

    // increment
    if(!(tag & 1) && page_bitmask_ptr){
      current_level_ptr++;
    }
    values_processed += run_num_values;
  }   // main work unit loop

  // warp 2 computed the validity count
  if(warp_id == 2 && warp_lane == 0){ 
    output_null_counts[page_idx] = num_rows[page_idx] - valid_count;            
  }
}


template <int num_warps_per_block>
__global__ void decode_pages_kernel5(const void* const* page_data,
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
  const int warp_id       = threadIdx.x / 32;  // warp id within a threadblock
  const int page_idx      = blockIdx.x;
  const int warp_lane     = threadIdx.x % 32;  

  const bool anchor_thread = warp_id == 0 && warp_lane == 0;  

  // page setup
  const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);
  // have to set this to 0 before we exit.
  if(anchor_thread){      
    output_null_counts[page_idx] = 0;    
  }

  // if we're out of range or if this has no data (dictionary pages will have a nullptr here)
  if(page_idx >= num_pages || page_data[page_idx] == nullptr){
    return;
  }  
  
  // all shared data
  /*
  typedef cub::WarpScan<uint8_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage;  
  */
  __shared__ dict_info di;

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
  uint8_t const* current_level_ptr = page_data_cur;
  auto level_start = current_level_ptr;
  page_data_cur += definition_level_size; 

  // dictionary handling
  uint8_t const* dict = dicts[page_idx];
  if(dict && anchor_thread){    
    di.dict_val = 0;
    di.dict_run = 0;
    di.dict_bits = *page_data_cur; 
    di.data_start = page_data_cur+1;
    di.data_end = page_start + page_size[page_idx];   
    di.dict_pos = 0;     
    di.read_pos = 0;
    di.dict_batch_len = 0;
  }

  // pointer to the start of the values section 
  const uint8_t* input_data_ptr          = page_data_cur;
  __shared__ int current_input_pos;
  if(anchor_thread){
    current_input_pos = 0;
  }
  uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
  cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
  cudf::size_type type_size = page_type_size[page_idx];
  int valid_count = 0;

  __shared__ int round_mask[2][2];
  __shared__ int round_count[2][2];
  __shared__ int round_output_index[2][32];

  __syncthreads();
  
  // loop {
  //   decode level data (should this be warpy?)
  //   loop { if no dictionary data, all values are processed on the first iteration
  //     warp 0 decodes dictionary data if necessary
  //     warp 1 decodes values
  //   }
  //   warp 2 decode validity
  // }   
  
  // Keep going until the definition levels have been completely parsed
  // Note that the end of the definition level section is the same as the start of the values
  // section
  int values_processed = 0;  
  while (reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)){
    // decode header stuff. should only 1 warp do this? I dunno.
    uint8_t  const tag         = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
    uint32_t const run_length  = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);
    // if this is a bitpacked run, there are 8 values for each byte in the level stream
    auto const     run_num_values = (tag & 1) ? run_length * 8 : run_length;

    // if we've got a dictionary, or bitpacked validity we have to process values in k5_loop_value_count chunks. otherwise
    // we can decode it all in one shot  
    // IMPORTANT: k5_loop_value_count must be a multiple of 8.      
    int num_values = (dict || (tag & 1)) ? min(k5_loop_value_count, run_num_values) : run_num_values;

    // warp 0 decodes the dictionary data (for the first iteration of the loop)
    if(dict || (tag & 1)){
      if(warp_id == 0){
        if(dict){
          // buffer up as much as we can.
          dict_buffer_to2(&di, current_input_pos + (k5_loop_value_count * 2), warp_lane);
        }
        // decode masks
        if(tag & 1){          
          compute_round_info(warp_lane, num_values, round_mask, round_count, round_output_index, current_level_ptr, 0);
        }
      }
      
      __syncthreads();
    }

    int run_values_processed = 0;
    int buf = 0;
    while(run_values_processed < run_num_values){
      // warp 0 decodes dictionary data (for the next iteration of the loop)
      if((dict || (tag & 1)) && warp_id == 0){
        if(dict){
          // note that we are only dictionary indices for non-null inputs.  So even
          // though the number of values in the level data (run_num_values) may be N, the number of dictionary
          // indices we read may be < N.
          dict_buffer_to2(&di, current_input_pos + (k5_loop_value_count * 2), warp_lane);        
        }
        if(tag & 1){
          compute_round_info(warp_lane, num_values, round_mask, round_count, round_output_index, current_level_ptr, !buf);
        }
      }

      // warp 1 decodes values
      if(warp_id == 1){
        int _current_input_pos = current_input_pos;
        auto _current_input_ptr = input_data_ptr + (_current_input_pos * type_size);

        // bitpacked run
        // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
        // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
        // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
        // null masks, we can directly copy from the bitpacked definition levels.
        if(tag & 1){          
          std::size_t num_rounds = utility::roundUpDiv(num_values, 32);
          for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {            
            std::size_t bit_idx         = round_idx * 32 + warp_lane;           
            auto const warp_aggregate = round_count[buf][0];
                        
            if (round_mask[buf][0] & (1 << warp_lane)) {
              auto const exclusive_output = round_output_index[buf][warp_lane];              

              auto const dict_pos = dict ? (_current_input_pos + exclusive_output) & (dict_buf_size - 1) : 0;
              auto const dict_val = dict ? di.dict_idx[dict_pos] : 0;
              auto const src = dict ? dict + (dict_val * type_size)
                                    : _current_input_ptr + exclusive_output * type_size;
              
              /*
              uint64_t output_pos = ((current_output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;              
              if(page_idx == 1 && output_pos < 256){                
                printf("COPYB0(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
              } 
              */               

              switch(type_size){
              case 4: copy_val4(current_output_ptr + (bit_idx * type_size), src); break;
              case 8: copy_val8(current_output_ptr + (bit_idx * type_size), src); break;
              default: break;
              }
            }

            _current_input_ptr += (warp_aggregate * type_size);
            _current_input_pos += warp_aggregate;
          }
        } 
        // RLE run
        // Again, for a flat data type, each value has 1 bit stored in the definition level: 0 or 1.
        // So the repeated value must be either 0 or 1. If the repeated value is 0, it means we have
        // repeated NULLs in the column. Since the NULL mask is initialized to 0, we do not need to
        // do anything. If the repeated value is 1, we need to copy the data, and set the null mask
        // to 1.
        //
        // For pages with no definition levels, pretend the validity value is just 1 and decode all
        // the values in 1 loop          
        else {
          uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *current_level_ptr;
          if (repeated_value) {
            uint32_t output_base_idx = 0;
            do {
              int warp_aggregate = output_base_idx + 32 > num_values ? num_values - output_base_idx : 32;
              
              if(output_base_idx + warp_lane < num_values){
                auto const output_idx = output_base_idx + warp_lane;

                auto const dict_pos = dict ? (_current_input_pos + warp_lane) & (dict_buf_size - 1) : 0;
                auto const dict_val = dict ? di.dict_idx[dict_pos] : 0;
                auto const src = dict ? dict + (dict_val * type_size)
                                      : _current_input_ptr + output_idx * type_size;

                /*
                uint64_t output_pos = ((current_output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;                
                if(page_idx == 1 && output_pos < 256){                  
                  printf("COPYB1(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
                }
                */

                switch(type_size){
                case 4: copy_val4(current_output_ptr + (output_idx * type_size), src); break;
                case 8: copy_val8(current_output_ptr + (output_idx * type_size), src); break;
                default: break;
                }
              }

              _current_input_pos += warp_aggregate;
              output_base_idx += warp_aggregate;
            } while(output_base_idx < num_values);
          }
        }

        // warp 0 needs to know how many actual non-null values we processed so it 
        // can buffer the dictionary appropriately
        if(warp_lane == 0){
          current_input_pos = _current_input_pos;
        }  
      }
      // warp 2 decodes validity
      else if(warp_id == 2){
        // bitpacked
        if(tag & 1){
          if(page_bitmask_ptr != nullptr){            
            valid_count += copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, current_bitmask_offset, current_level_ptr, num_values/8, warp_id);
          } else {
            printf("THIS SHOULDNT HAPPEN\n");
          }
        }
        // repeated
        else {
          // if this is not a nullable column or if the repeated value is 1 all the values are valid
          bool const valid = page_bitmask_ptr ? *current_level_ptr : true;
          if(valid){
            if(page_bitmask_ptr){
              set_validity_bits_safe(page_bitmask_ptr, current_bitmask_offset, num_values);
            }
            valid_count += num_values;
          }
        }
      }
      
      // everyone increments      
      current_output_ptr += (num_values * type_size); 
      current_bitmask_offset += num_values;
      if((tag & 1) && page_bitmask_ptr){        
        // 1 bit per value, so 8 values per byte.
        current_level_ptr += num_values / 8;
      } 
      run_values_processed += num_values;

      // next batch of values
      num_values = (dict || (tag & 1)) ? min(k5_loop_value_count, run_num_values - run_values_processed) : run_num_values;
     
      // loop sync
      buf = !buf;
      __syncthreads();
    } // inner value decoding loop

    // increment
    if(!(tag & 1) && page_bitmask_ptr){
      current_level_ptr++;
    }
    values_processed += run_num_values;
  }   // main work unit loop

  // warp 2 computed the validity count
  if(warp_id == 2 && warp_lane == 0){ 
    output_null_counts[page_idx] = num_rows[page_idx] - valid_count;            
  }
}

template <int num_warps_per_block>
__global__ void decode_pages_kernel4(const void* const* page_data,
                                    const std::size_t* page_size,
                                    uint8_t* const* output_data,
                                    cudf::bitmask_type* const* output_bitmask,
                                    const cudf::size_type* output_bitmask_offset,
                                    const cudf::size_type* page_type_size,
                                    uint8_t const** dicts,
                                    cudf::size_type const* num_rows,
                                    cudf::size_type* output_null_counts,
                                    int const *dict_page_indices,
                                    std::size_t num_pages)
{  
  const int warp_id       = threadIdx.x / 32;  // warp id within a threadblock
  const int page_idx      = blockIdx.x;
  const int warp_lane     = threadIdx.x % 32;  

  const bool anchor_thread = warp_id == 0 && warp_lane == 0;  

  // page setup
  const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);
  // have to set this to 0 before we exit.
  if(anchor_thread){      
    output_null_counts[page_idx] = 0;    
  }

  // if we're out of range or if this has no data (dictionary pages will have a nullptr here)
  if(page_idx >= num_pages || page_data[page_idx] == nullptr){
    return;
  }  
  
  // all shared data
  typedef cub::WarpScan<uint8_t> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage;  
  __shared__ dict_info di;  

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
  uint8_t const* current_level_ptr = page_data_cur;
  auto level_start = current_level_ptr;
  page_data_cur += definition_level_size; 

  // dictionary handling
  uint8_t const* dict = dicts[page_idx];
  if(dict && anchor_thread){    
    di.dict_val = 0;
    di.dict_run = 0;
    di.dict_bits = *page_data_cur; 
    di.data_start = page_data_cur+1;
    di.data_end = page_start + page_size[page_idx];   
    di.dict_pos = 0;     
    di.read_pos = 0;
    di.dict_batch_len = 0;
  }

  // pointer to the start of the values section 
  const uint8_t* input_data_ptr          = page_data_cur;
  __shared__ int current_input_pos;
  if(anchor_thread){
    current_input_pos = 0;
  }
  uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
  // cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
  cudf::size_type current_bitmask_offset = 0;
  cudf::size_type type_size = page_type_size[page_idx];
  int valid_count = page_bitmask_ptr == nullptr ? num_rows[page_idx] : 0;
  
  __syncthreads();
  
  // loop {
  //   decode level data (should this be warpy?)
  //   loop { if no dictionary data, all values are processed on the first iteration
  //     warp 0 decodes dictionary data if necessary
  //     warp 1 decodes values
  //   }
  //   warp 2 decode validity
  // }    

  // Keep going until the definition levels have been completely parsed
  // Note that the end of the definition level section is the same as the start of the values
  // section
  int values_processed = 0;
  while ((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)) || (values_processed < num_rows[page_idx])){
    // decode header stuff. should only 1 warp do this? I dunno.
    uint8_t  const tag         = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
    uint32_t const run_length  = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);
    // if this is a bitpacked run, there are 8 values for each byte in the level stream
    auto const     run_num_values = (tag & 1) ? run_length * 8 : run_length;

    // if we've got a dictionary, we have to process values in dict_buf_count chunks. otherwise
    // we can decode it all in one shot  
    // IMPORTANT: dict_buf_count must be a multiple of 8.      
    int num_values = dict ? min(dict_buf_count, run_num_values) : run_num_values;

    // warp 0 decodes the dictionary data (for the first iteration of the loop)
    if(dict){
      if(warp_id == 0){
        // note that we are only dictionary indices for non-null inputs.  So even
        // though the number of values in the level data (run_num_values) may be N, the number of dictionary
        // indices we read may be < N.
        dict_buffer_to2(&di, current_input_pos + (dict_buf_count * 2), warp_lane);      
      }
      __syncthreads();
    }

    int run_values_processed = 0;
    while(run_values_processed < run_num_values){      
      // warp 0 decodes dictionary data (for the next iteration of the loop)
      if(dict && warp_id == 0){
        // note that we are only dictionary indices for non-null inputs.  So even
        // though the number of values in the level data (run_num_values) may be N, the number of dictionary
        // indices we read may be < N.
        dict_buffer_to2(&di, current_input_pos + (dict_buf_count * 2), warp_lane);        
      }      

      // warp 1 decodes values
      if(warp_id == 1){
        int _current_input_pos = current_input_pos;
        auto _current_input_ptr = input_data_ptr + (_current_input_pos * type_size);

        // bitpacked run
        // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
        // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
        // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
        // null masks, we can directly copy from the bitpacked definition levels.
        if(tag & 1){          
          std::size_t num_rounds = utility::roundUpDiv(num_values, 32);
          for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
            std::size_t bit_idx         = round_idx * 32 + warp_lane;
            std::size_t byte_idx        = bit_idx / 8;
            std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

            uint8_t mask             = 0;
            uint8_t exclusive_output = 0; // output position per thread
            uint8_t warp_aggregate   = 0; // total # of values decoded for the whole warp

            if (byte_idx < (num_values / 8)) {
              uint8_t current_byte = current_level_ptr[byte_idx];
              // mask will be either 0 or 1, indicating whether the current value is NULL
              mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
            }

            WarpScan(temp_storage).ExclusiveSum(mask, exclusive_output, warp_aggregate);
            
            if (mask) {
              auto const dict_pos = dict ? (_current_input_pos + exclusive_output) & (dict_buf_size - 1) : 0;
              auto const dict_val = dict ? di.dict_idx[dict_pos] : 0;                                                        
              auto const dict_ptr = dict; 

              auto const src = dict ? dict_ptr + (dict_val * type_size)
                                    : _current_input_ptr + exclusive_output * type_size;              

              /*   
              // printf("I: %d, %d, %d\n", dict_val, bit_idx, page_idx);              
              uint64_t output_pos = ((current_output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;              
              if(page_idx == 0 && output_pos < 256){                
                printf("COPYB0(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
              }
              */
              switch(type_size){
              case 4: copy_val4(current_output_ptr + (bit_idx * type_size), src); break;
              case 8: copy_val8(current_output_ptr + (bit_idx * type_size), src); break;
              default: break;
              }
            }

            _current_input_ptr += (warp_aggregate * type_size);
            _current_input_pos += warp_aggregate;
          }
        } 
        // RLE run
        // Again, for a flat data type, each value has 1 bit stored in the definition level: 0 or 1.
        // So the repeated value must be either 0 or 1. If the repeated value is 0, it means we have
        // repeated NULLs in the column. Since the NULL mask is initialized to 0, we do not need to
        // do anything. If the repeated value is 1, we need to copy the data, and set the null mask
        // to 1.
        //
        // For pages with no definition levels, pretend the validity value is just 1 and decode all
        // the values in 1 loop          
        else {
          uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *current_level_ptr;
          if (repeated_value) {
            uint32_t output_base_idx = 0;
            do {
              int warp_aggregate = output_base_idx + 32 > num_values ? num_values - output_base_idx : 32;
              
              if(output_base_idx + warp_lane < num_values){
                auto const output_idx = output_base_idx + warp_lane;

                auto const dict_pos = dict ? (_current_input_pos + warp_lane) & (dict_buf_size - 1) : 0;
                auto const dict_val = dict ? di.dict_idx[dict_pos] : 0;                
                auto const dict_ptr = dict;                
                auto const src = dict ? dict_ptr + (dict_val * type_size)
                                      : _current_input_ptr + output_idx * type_size;

                /*          
                uint64_t output_pos = ((current_output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;                
                if(page_idx == 0 && output_pos < 256){                  
                  printf("COPYB1(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
                } 
                */               
                // printf("I: %d, %d, %d\n", dict_val, output_idx, page_idx);
                switch(type_size){
                case 4: copy_val4(current_output_ptr + (output_idx * type_size), src); break;
                case 8: copy_val8(current_output_ptr + (output_idx * type_size), src); break;
                default: break;
                } 
              }

              _current_input_pos += warp_aggregate;
              output_base_idx += warp_aggregate;
            } while(output_base_idx < num_values);
          }
        }

        // warp 0 needs to know how many actual non-null values we processed so it 
        // can buffer the dictionary appropriately
        if(warp_lane == 0){
          current_input_pos = _current_input_pos;
        }  
      }
      // warp 2 decodes validity
      else if(warp_id == 2){
        // bitpacked
        if(tag & 1){
          if(page_bitmask_ptr != nullptr){            
            valid_count += copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, current_bitmask_offset, output_bitmask_offset[page_idx], num_rows[page_idx], current_level_ptr, num_values/8, warp_id);
          } else {
            printf("THIS SHOULDNT HAPPEN\n");
          }
        }
        // repeated
        else {
          // if this is not a nullable column or if the repeated value is 1 all the values are valid
          bool const valid = page_bitmask_ptr ? *current_level_ptr : true;
          if(valid){
            if(page_bitmask_ptr){
              set_validity_bits_safe(page_bitmask_ptr, current_bitmask_offset + output_bitmask_offset[page_idx], num_values);
            }
            valid_count += num_values;
          }
        }
      }
      
      // everyone increments      
      current_output_ptr += (num_values * type_size); 
      current_bitmask_offset += num_values;
      if((tag & 1) && page_bitmask_ptr){        
        // 1 bit per value, so 8 values per byte.
        current_level_ptr += num_values / 8;
      } 
      run_values_processed += num_values;

      // next batch of values
      num_values = dict ? min(dict_buf_count, run_num_values - run_values_processed) : run_num_values;
     
      __syncthreads();
    } // inner value decoding loop

    // increment
    if(!(tag & 1) && page_bitmask_ptr){
      current_level_ptr++;
    }
    values_processed += run_num_values;
  }   // main work unit loop

  // warp 2 computed the validity count
  if(warp_id == 2 && warp_lane == 0){ 
    // printf("PNULL COUNT(%d) : %d\n", page_idx, num_rows[page_idx] - valid_count);
    output_null_counts[page_idx] = num_rows[page_idx] - valid_count;            
  }
}


// optimization oppurtunities:
//
// - for non-nullable fixed-width pages that are not dictionary-based, the decode is essentially a memcpy. a seperate kernel
//   that parallelizes
//
template <int num_warps_per_block>
__launch_bounds__(num_warps_per_block * 32) __global__
__global__ void decode_pages_kernel3(const void* const* page_data,
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

  // const int warp_id       = blockIdx.x * num_warps_per_block + threadIdx.x / 32;
  const int page_idx      = blockIdx.x;
  const int local_warp_id = threadIdx.x / 32;  // warp id within a threadblock
  // const int num_warps     = gridDim.x * num_warps_per_block;
  const int warp_lane     = threadIdx.x % 32;

  const bool anchor_thread = local_warp_id == 0 && warp_lane == 0;  

  // printf("WHEE\n");

  if(page_idx >= num_pages){
    return;
  }    

  typedef cub::WarpScan<cudf::size_type> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[num_warps_per_block];  
  // __shared__ dict_info block_dict_info[num_warps_per_block];    
  __shared__ work_unit work[num_warps_per_block];  
  __shared__ cudf::size_type values_processed; 

  // page setup
  const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);
  if(anchor_thread){      
    output_null_counts[page_idx] = 0;
    values_processed = 0;    
  }
  
  // dictionary pages have nothing to decode.
  if(page_start == nullptr){
    return;
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

  // dictionary handling
  uint8_t const* dict = dicts[page_idx];

  // pointer to the definition levels currently being decoded
  // const uint8_t* current_level_ptr = page_data_cur;     
  __shared__ uint8_t const* current_level_ptr;
  if(anchor_thread){
    current_level_ptr = page_data_cur;     
  }
  __syncthreads();
  auto level_start = current_level_ptr;
  page_data_cur += definition_level_size;

  // running position of the dictionary.
  dict_info dict_cur;
  if(dict && anchor_thread){    
    dict_cur.dict_val = 0;
    dict_cur.dict_run = 0;
    dict_cur.dict_bits = *page_data_cur; 
    dict_cur.data_start = page_data_cur+1;
    dict_cur.data_end = page_start + page_size[page_idx];   
    dict_cur.dict_pos = 0;     
    dict_cur.read_pos = 0;
    dict_cur.dict_batch_len = 0;
  }

  // pointer to the start of the values section 
  const uint8_t* input_data_ptr          = page_data_cur;
  const uint8_t* current_data_ptr        = input_data_ptr;
  uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
  cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
  cudf::size_type type_size = page_type_size[page_idx];
  cudf::size_type thread_valid_count = 0;  

  __shared__ rle3_run runs[num_warps_per_block];
  __shared__ rle3_stream stream;
  if(anchor_thread){
    stream.cur = current_level_ptr;
    stream.end = input_data_ptr;
  }  
  
  __syncthreads();

  int wu_index = 0;      

  // reminder:  this check doesn't work for list types, since the number of rows isn't necessarily the same as the number
  // of values.  they can span pages.
  while((reinterpret_cast<uintptr_t>(stream.cur) < reinterpret_cast<uintptr_t>(stream.end)) /*|| (values_processed < num_rows[page_idx])*/){    
    // step 1, generate the next set of work units.
    int num_runs = generate_rle3_runs<num_warps_per_block>(runs, num_warps_per_block, stream);
    if(local_warp_id == 0 && warp_lane < num_warps_per_block){
      work[warp_lane].length = 0;
    }
    if(local_warp_id == 0 && warp_lane < num_runs){      
      work_unit *w = &work[warp_lane];
      rle3_run const* r = &runs[warp_lane];

      w->tag = r->tag;
      w->length = r->length;
      w->level_ptr = r->data;
      
      /*
      w->bitmask_offset = current_bitmask_offset; 
      w->output_ptr = current_output_ptr;
      // increment
      if(tag & 1){
        current_output_ptr += (length * 8 * type_size); 
        current_bitmask_offset += (length * 8);
        values_processed += (length * 8);
        if(page_bitmask_ptr){
          current_level_ptr += length;
        }            
      } else {
        current_output_ptr += (length * type_size);
        current_bitmask_offset += length;
        values_processed += length;
        if(page_bitmask_ptr){
          current_level_ptr++;
        }
      }
      */
    }         
    
    __syncthreads();

    continue;

    work_unit *w = &work[local_warp_id];    

    // step 2, generate the offsets into the input data for each work unit to read.
    if(w->length > 0){
      if(warp_lane == 0){        
        w->input_pos = 0;       
      }
      
      // rle encoded run. we will have to sift through all the values
      if(w->tag & 1){
        std::size_t num_rounds = utility::roundUpDiv(w->length * 8, 32);        
        for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
          std::size_t bit_idx         = round_idx * 32 + warp_lane;
          std::size_t byte_idx        = bit_idx / 8;
          std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

          uint8_t mask             = 0;
          cudf::size_type exclusive_output = 0; // output position per thread
          cudf::size_type warp_aggregate   = 0; // total # of values decoded for the whole warp

          if (byte_idx < w->length) {
            uint8_t current_byte = w->level_ptr[byte_idx];
            // mask will be either 0 or 1, indicating whether the current value is NULL
            mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
          }

          WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);
          if(warp_lane == 0){
            w->input_pos += warp_aggregate;
          }
        }
      } 
      // the simple case - just a repeated value (for non-nullable columns we always go through here)      
      else {
        if(warp_lane == 0){
          uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *w->level_ptr;
          w->input_pos += repeated_value ? w->length : 0;
        }
      }
    }
    __syncthreads();        

    // generate offsets
    if(local_warp_id == 0){
      cudf::size_type w_pos = warp_lane < num_warps_per_block ? w[warp_lane].input_pos : 0;
      cudf::size_type warp_aggregate;
      WarpScan(temp_storage[0]).ExclusiveScan(w_pos, w_pos, cudf::size_type{0}, cub::Sum(), warp_aggregate);      
      if(warp_lane < num_warps_per_block){
        // absolute position
        w[warp_lane].input_pos = w_pos + ((current_data_ptr - page_data_cur) / type_size);
      }      
      current_data_ptr += (warp_aggregate * type_size);

      // if we have dictionary data, generate it now      
      #if 0
      if(dict && warp_lane == 0){
        for(int idx=0; idx<num_warps_per_block; idx++){
          work_unit *w = &work[idx];
          if(w->length <= 0){
            break;
          }          
          /*
          if(w->input_pos < 256 && page_idx == 1){
            printf("WU %d (length %d)\n", idx, w->tag & 1 ? w->length * 8 : w->length);
          }          
          */
          dict_advance_to(&dict_cur, w->input_pos, page_idx);   
          w->dict.copy_pos(dict_cur);
        }
      }
      #endif
    }
    __syncthreads();
    
    /*
    if(local_warp_id == 0 && page_idx == 83 && warp_lane == 0){
      for(int idx=0; idx<num_warps_per_block; idx++){
        if(w[idx].length > 0){
          uint64_t level_dist = w[idx].level_ptr - level_start;
          printf("WU(%d): %d, %d(%d), %d, %lu\n", wu_index, w[idx].length, (int)w[idx].tag, w[idx].tag & 1 ? 1 : 0, w[idx].input_pos, level_dist);

          wu_index++;
        }
      }
    }
    __syncthreads();    

    return;
    */
       
    // process the work units. 1 warp == 1 work unit    
    if(w->length > 0){
      // process the work units
      if (w->tag & 1) {
        // bitpacked run
        // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
        // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
        // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
        // null masks, we can directly copy from the bitpacked definition levels.
        std::size_t num_rounds = utility::roundUpDiv(w->length * 8, 32);
        int input_pos = w->input_pos;
        for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
          std::size_t bit_idx         = round_idx * 32 + warp_lane;
          std::size_t byte_idx        = bit_idx / 8;
          std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

          uint8_t mask             = 0;
          cudf::size_type exclusive_output = 0; // output position per thread
          cudf::size_type warp_aggregate   = 0; // total # of values decoded for the whole warp

          if (byte_idx < w->length) {
            uint8_t current_byte = w->level_ptr[byte_idx];
            // mask will be either 0 or 1, indicating whether the current value is NULL
            mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
          }

          WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);
          
          #if 0
          if(dict){
            // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
            dict_buffer_to(&w->dict, input_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
          }
                    
          if (mask) {            
            auto const dict_pos = (input_pos + exclusive_output) & (dict_buf_size - 1);
            auto const dict_val = w->dict.dict_idx[dict_pos];        
            auto const src = dict ? dict + (dict_val * type_size)
                                  : (input_data_ptr + (input_pos * type_size)) + (exclusive_output * type_size);
                                  
            // auto const src = (input_data_ptr + (input_pos * type_size)) + (exclusive_output * type_size);
            auto const output_pos = ((w->output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;
            /*
            if(page_idx == 1 && output_pos < 256 && local_warp_id == 1){
              printf("COPYA(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", local_warp_id, output_pos, dict_pos, dict_val);
            }
            */
            switch(type_size){
            case 4: copy_val4(w->output_ptr + (bit_idx * type_size), src); break;
            case 8: copy_val8(w->output_ptr + (bit_idx * type_size), src); break;
            default: break;
            }

            thread_valid_count++;
          }
          #endif

          input_pos += warp_aggregate;
        }
        
        #if 0
        if(page_bitmask_ptr != nullptr){
          copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, w->bitmask_offset, w->level_ptr, w->length, local_warp_id);
        }
        #endif
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
        uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *w->level_ptr;

        if (repeated_value) {
          uint32_t output_base_idx = 0;
          int input_pos = w->input_pos;
          do {
            int warp_aggregate = output_base_idx + 32 > w->length ? w->length - output_base_idx : 32;
            
            #if 0
            if(dict){
              // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
              dict_buffer_to(&w->dict, input_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
            }

            if(output_base_idx + warp_lane < w->length){    
              auto const dict_pos = (input_pos + warp_lane) & (dict_buf_size - 1);
              auto const dict_val = w->dict.dict_idx[dict_pos];
              auto const output_idx = output_base_idx + warp_lane;
              auto const src = dict ? dict + (dict_val * type_size)
                                    : (input_data_ptr + (input_pos * type_size)) + output_idx * type_size;

              auto const output_pos = ((w->output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;
              /*
              if(page_idx == 1 && output_pos < 256 && local_warp_id == 1){
                printf("COPYB(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", local_warp_id, output_pos, dict_pos, dict_val);
              }
              */
              switch(type_size){              
              case 4: copy_val4(w->output_ptr + (output_idx * type_size), src); break;
              case 8: copy_val8(w->output_ptr + (output_idx * type_size), src); break;
              default: break;
              }

              thread_valid_count++;
            }   
            #endif         

            input_pos += warp_aggregate;
            output_base_idx += warp_aggregate;            
          } while(output_base_idx < w->length);
          
          #if 0
          if(page_bitmask_ptr != nullptr){
            set_validity_bits_safe(page_bitmask_ptr, w->bitmask_offset, w->length);
          }
          #endif
        }
      }      
    }       

    __syncthreads();
  }

  // compute validity count if applicable
  if(page_bitmask_ptr){    
    using BlockReduce = cub::BlockReduce<cudf::size_type, num_warps_per_block * 32>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    cudf::size_type valid_count = BlockReduce(temp_storage).Sum(thread_valid_count);
    if(anchor_thread){
      output_null_counts[page_idx] = num_rows[page_idx] - valid_count;
    }
  }
}


// optimization oppurtunities:
//
// - for non-nullable fixed-width pages that are not dictionary-based, the decode is essentially a memcpy. a seperate kernel
//   that parallelizes
//
#if 0
template <int num_warps_per_block>
__launch_bounds__(num_warps_per_block * 32) __global__
__global__ void decode_pages_kernel3(const void* const* page_data,
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

  // const int warp_id       = blockIdx.x * num_warps_per_block + threadIdx.x / 32;
  const int page_idx      = blockIdx.x;
  const int local_warp_id = threadIdx.x / 32;  // warp id within a threadblock
  // const int num_warps     = gridDim.x * num_warps_per_block;
  const int warp_lane     = threadIdx.x % 32;

  const bool anchor_thread = local_warp_id == 0 && warp_lane == 0;

  if(page_idx >= num_pages){
    return;
  }    

  typedef cub::WarpScan<cudf::size_type> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[num_warps_per_block];  
  // __shared__ dict_info block_dict_info[num_warps_per_block];    
  __shared__ work_unit work[num_warps_per_block];  
  __shared__ cudf::size_type values_processed; 

  // page setup
  const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);
  if(anchor_thread){      
    output_null_counts[page_idx] = 0;
    values_processed = 0;    
  }
  
  // dictionary pages have nothing to decode.
  if(page_start == nullptr){
    return;
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

  // dictionary handling
  uint8_t const* dict = dicts[page_idx];

  // pointer to the definition levels currently being decoded
  // const uint8_t* current_level_ptr = page_data_cur;     
  __shared__ uint8_t const* current_level_ptr;
  if(anchor_thread){
    current_level_ptr = page_data_cur;     
  }
  page_data_cur += definition_level_size;

  // running position of the dictionary.
  dict_info dict_cur;
  if(dict && anchor_thread){    
    dict_cur.dict_val = 0;
    dict_cur.dict_run = 0;
    dict_cur.dict_bits = *page_data_cur; 
    dict_cur.data_start = page_data_cur+1;
    dict_cur.data_end = page_start + page_size[page_idx];   
    dict_cur.dict_pos = 0;     
    dict_cur.read_pos = 0;
    dict_cur.dict_batch_len = 0;
  }

  // pointer to the start of the values section 
  const uint8_t* input_data_ptr          = page_data_cur;
  const uint8_t* current_data_ptr        = input_data_ptr;
  uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
  cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
  cudf::size_type type_size = page_type_size[page_idx];
  cudf::size_type thread_valid_count = 0;

  __syncthreads();

  // debug junk
  int dict_advances = 0;
  int total_work_units_processed = 0;
  int wu_index = 0;

  auto const max_work_units = num_warps_per_block - 1;
  
  // reminder:  this check doesn't work for list types, since the number of rows isn't necessarily the same as the number
  // of values.  they can span pages.  
  // while(values_processed < num_rows[page_idx]){
  while((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)) || (values_processed < num_rows[page_idx])){
    // warp 0 decodes the next set of work units
    if(local_warp_id == 0){      
      // zero the length on every work unit.  potentially not every warp will actually get something to do.
      if(warp_lane < max_work_units){
        work[warp_lane].length = 0;
      }
      // buffer up a number of work units (runs to decode). the level definition encoding makes this pretty much impossible
      // to parallelize, unfortunately, so we rely on one thread. ugh.
      if(warp_lane == 0){
        int work_units_processed = 0; 
        while (((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)) || (values_processed < num_rows[page_idx]))
                && 
                work_units_processed < max_work_units){          
          
          uint8_t tag     = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
          uint32_t length = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);
          
          work_unit *w = &work[work_units_processed++];
          w->tag = tag;
          w->length = length;
          w->bitmask_offset = current_bitmask_offset;
          w->level_ptr = current_level_ptr;
          w->output_ptr = current_output_ptr;

          // increment
          if(tag & 1){
            current_output_ptr += (length * 8 * type_size); 
            current_bitmask_offset += (length * 8);
            values_processed += (length * 8);
            if(page_bitmask_ptr){
              current_level_ptr += length;
            }            
          } else {
            current_output_ptr += (length * type_size);
            current_bitmask_offset += length;
            values_processed += length;
            if(page_bitmask_ptr){
              current_level_ptr++;
            }
          }          
        }      

        total_work_units_processed += work_units_processed;  
      }    
      __syncwarp();
    }
    // all other warps decode
    else {
      auto const work_id = local_warp_id-1;
      work_unit *w = &work[work_id];

      // step 2, generate the offsets into the input data for each work unit to read.
      if(w->length > 0){
        if(warp_lane == 0){
          w->input_pos = 0;
        }
        
        // rle encoded run. we will have to sift through all the values
        if(w->tag & 1){
          std::size_t num_rounds = utility::roundUpDiv(w->length * 8, 32);        
          for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
            std::size_t bit_idx         = round_idx * 32 + warp_lane;
            std::size_t byte_idx        = bit_idx / 8;
            std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

            uint8_t mask             = 0;
            cudf::size_type exclusive_output = 0; // output position per thread
            cudf::size_type warp_aggregate   = 0; // total # of values decoded for the whole warp

            if (byte_idx < w->length) {
              uint8_t current_byte = w->level_ptr[byte_idx];
              // mask will be either 0 or 1, indicating whether the current value is NULL
              mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
            }

            WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);
            if(warp_lane == 0){
              w->input_pos += warp_aggregate;
            }
          }
        }
        // the simple case - just a repeated value (for non-nullable columns we always go through here)
        else {
          if(warp_lane == 0){
            uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *w->level_ptr;
            w->input_pos += repeated_value ? w->length : 0;
          }
        }
      }

      // we need to wait for all warps to be complete at this point so that input_length for all work units
      // is complete.
      __syncthreads();

      // generate offsets for all work
      if(local_warp_id == 1){
        cudf::size_type w_pos = warp_lane < num_warps_per_block ? work[warp_lane].input_pos : 0;
        cudf::size_type warp_aggregate;
        WarpScan(temp_storage[1]).ExclusiveScan(w_pos, w_pos, cudf::size_type{0}, cub::Sum(), warp_aggregate);
        if(warp_lane < num_warps_per_block){
          // absolute position
          work[warp_lane].input_pos = w_pos + ((current_data_ptr - page_data_cur) / type_size);
        }     
        current_data_ptr += (warp_aggregate * type_size);
      }

      // done. now back to multiple warps running
      __syncthreads();

      // if we have dictionary data, generate it now                  
      if(dict && warp_lane == 0){
        for(int idx=0; idx<num_warps_per_block; idx++){
          work_unit *w = &work[idx];
          if(w->length <= 0){
            break;
          }
          dict_advance_to(&dict_cur, w->input_pos, page_idx);
          w->dict.copy_pos(dict_cur);
          dict_advances++;
        }
      }      
      __syncwarp();
      
      if(local_warp_id == 0 && page_idx == 1 && warp_lane == 0){      
        for(int idx=0; idx<num_warps_per_block; idx++){
          if(w[idx].length > 0 && wu_index < 2048){
            uint64_t level_dist = w[idx].level_ptr - page_start;
            printf("WU(%d): %d, %d, %d, %d, %lu\n", wu_index, total_work_units_processed, w[idx].length, (int)w[idx].tag, w[idx].input_pos, level_dist);
            wu_index++;
          }
        }
      }
      __syncthreads();
    }
                
      work_unit *w = &work[local_warp_id-1];

      // process the work units. 1 warp == 1 work unit    
      if(w->length > 0){
        // process the work units
        if (w->tag & 1) {
          // bitpacked run
          // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
          // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
          // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
          // null masks, we can directly copy from the bitpacked definition levels.
          std::size_t num_rounds = utility::roundUpDiv(w->length * 8, 32);
          int input_pos = w->input_pos;
          for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
            std::size_t bit_idx         = round_idx * 32 + warp_lane;
            std::size_t byte_idx        = bit_idx / 8;
            std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

            uint8_t mask             = 0;
            cudf::size_type exclusive_output = 0; // output position per thread
            cudf::size_type warp_aggregate   = 0; // total # of values decoded for the whole warp

            if (byte_idx < w->length) {
              uint8_t current_byte = w->level_ptr[byte_idx];
              // mask will be either 0 or 1, indicating whether the current value is NULL
              mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
            }

            WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);
                      
            if(dict){
              // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
              dict_buffer_to(&w->dict, input_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
            }
                                
            if (mask) {            
              auto const dict_pos = (input_pos + exclusive_output) & (dict_buf_size - 1);
              auto const dict_val = w->dict.dict_idx[dict_pos];        
              auto const src = dict ? dict + (dict_val * type_size)
                                    : (input_data_ptr + (input_pos * type_size)) + (exclusive_output * type_size);
                                    
              // auto const src = (input_data_ptr + (input_pos * type_size)) + (exclusive_output * type_size);
              auto const output_pos = ((w->output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;
              /*
              if(page_idx == 1 && output_pos < 256 && local_warp_id == 1){
                printf("COPYA(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", local_warp_id, output_pos, dict_pos, dict_val);
              }
              */
              switch(type_size){
              case 4: copy_val4(w->output_ptr + (bit_idx * type_size), src); break;
              case 8: copy_val8(w->output_ptr + (bit_idx * type_size), src); break;
              default: break;
              }

              thread_valid_count++;
            } 

            input_pos += warp_aggregate;
          }        
          
          if(page_bitmask_ptr != nullptr){
            copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, w->bitmask_offset, w->level_ptr, w->length, local_warp_id);
          }
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
          uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *w->level_ptr;

          if (repeated_value) {
            uint32_t output_base_idx = 0;
            int input_pos = w->input_pos;
            do {
              int warp_aggregate = output_base_idx + 32 > w->length ? w->length - output_base_idx : 32;
                          
              if(dict){
                // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
                dict_buffer_to(&w->dict, input_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
              }

              if(output_base_idx + warp_lane < w->length){    
                auto const dict_pos = (input_pos + warp_lane) & (dict_buf_size - 1);
                auto const dict_val = w->dict.dict_idx[dict_pos];
                auto const output_idx = output_base_idx + warp_lane;
                auto const src = dict ? dict + (dict_val * type_size)
                                      : (input_data_ptr + (input_pos * type_size)) + output_idx * type_size;

                auto const output_pos = ((w->output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;
                /*
                if(page_idx == 1 && output_pos < 256 && local_warp_id == 1){
                  printf("COPYB(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", local_warp_id, output_pos, dict_pos, dict_val);
                }
                */
                switch(type_size){              
                case 4: copy_val4(w->output_ptr + (output_idx * type_size), src); break;
                case 8: copy_val8(w->output_ptr + (output_idx * type_size), src); break;
                default: break;
                }

                thread_valid_count++;
              }            

              input_pos += warp_aggregate;
              output_base_idx += warp_aggregate;            
            } while(output_base_idx < w->length);          
            
            if(page_bitmask_ptr != nullptr){
              set_validity_bits_safe(page_bitmask_ptr, w->bitmask_offset, w->length);
            }
          }
        }      
      }
    }
    
    __syncthreads();
  }

  // compute validity count if applicable
  if(page_bitmask_ptr){    
    using BlockReduce = cub::BlockReduce<cudf::size_type, num_warps_per_block * 32>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    cudf::size_type valid_count = BlockReduce(temp_storage).Sum(thread_valid_count);
    if(anchor_thread){
      output_null_counts[page_idx] = num_rows[page_idx] - valid_count;
    }
  }

  /*
  if(local_warp_id == 0 && warp_lane == 0){
    printf("Calls(%d): calculate_run_length(%d), advance_to(%d), num_rows(%d)\n", page_idx, total_work_units_processed, dict_advances, num_rows[page_idx]);
  }
  */  
}
#endif

// optimization oppurtunities:
//
// - for non-nullable fixed-width pages that are not dictionary-based, the decode is essentially a memcpy. a seperate kernel
//   that parallelizes
//
template <int num_warps_per_block>
__launch_bounds__(num_warps_per_block * 32) __global__
__global__ void decode_pages_kernel2(const void* const* page_data,
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

  // const int warp_id       = blockIdx.x * num_warps_per_block + threadIdx.x / 32;
  const int page_idx      = blockIdx.x;
  const int local_warp_id = threadIdx.x / 32;  // warp id within a threadblock
  // const int num_warps     = gridDim.x * num_warps_per_block;
  const int warp_lane     = threadIdx.x % 32;

  const bool anchor_thread = local_warp_id == 0 && warp_lane == 0;  

  // printf("WHEE\n");

  if(page_idx >= num_pages){
    return;
  }    

  typedef cub::WarpScan<cudf::size_type> WarpScan;
  __shared__ typename WarpScan::TempStorage temp_storage[num_warps_per_block];  
  // __shared__ dict_info block_dict_info[num_warps_per_block];    
  __shared__ work_unit work[num_warps_per_block];  
  __shared__ cudf::size_type values_processed; 

  // page setup
  const uint8_t* page_start            = static_cast<const uint8_t*>(page_data[page_idx]);
  if(anchor_thread){      
    output_null_counts[page_idx] = 0;
    values_processed = 0;    
  }
  
  // dictionary pages have nothing to decode.
  if(page_start == nullptr){
    return;
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

  // dictionary handling
  uint8_t const* dict = dicts[page_idx];

  // pointer to the definition levels currently being decoded
  // const uint8_t* current_level_ptr = page_data_cur;     
  __shared__ uint8_t const* current_level_ptr;  
  if(anchor_thread){
    current_level_ptr = page_data_cur;     
  }
  __syncthreads();
  auto level_start = current_level_ptr;
  page_data_cur += definition_level_size;

  // running position of the dictionary.
  dict_info dict_cur;
  if(dict && anchor_thread){    
    dict_cur.dict_val = 0;
    dict_cur.dict_run = 0;
    dict_cur.dict_bits = *page_data_cur; 
    dict_cur.data_start = page_data_cur+1;
    dict_cur.data_end = page_start + page_size[page_idx];   
    dict_cur.dict_pos = 0;     
    dict_cur.read_pos = 0;
    dict_cur.dict_batch_len = 0;
  }

  // pointer to the start of the values section 
  const uint8_t* input_data_ptr          = page_data_cur;
  const uint8_t* current_data_ptr        = input_data_ptr;
  uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
  cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
  cudf::size_type type_size = page_type_size[page_idx];
  cudf::size_type thread_valid_count = 0;

  __syncthreads();
  
  int wu_index = 0;  

  // reminder:  this check doesn't work for list types, since the number of rows isn't necessarily the same as the number
  // of values.  they can span pages.  
  // while(values_processed < num_rows[page_idx]){
  while((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)) /*|| (values_processed < num_rows[page_idx])*/){
    // step 1, generate the next set of work units. warp 0 does all the work.
    if(local_warp_id == 0){   
      // zero the length on every work unit.  potentially not every warp will actually get something to do.
      if(warp_lane < num_warps_per_block){
        work[warp_lane].length = 0;
      }

      // buffer up a number of work units (runs to decode)      
      if(warp_lane == 0){
        int work_units_processed = 0; 
        while (((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr))/* || (values_processed < num_rows[page_idx])*/)
                && 
                work_units_processed < num_warps_per_block){

          uint64_t level_dist = current_level_ptr - level_start;                    
          
          uint8_t tag     = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
          uint32_t length = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);                    
          // printf("DIST: %lu, length(%d), tag(%d)\n", level_dist, tag & 1 ? length * 8 : length, (int)tag);
          
          work_unit *w = &work[work_units_processed++];
          w->tag = tag;
          w->length = length;
          w->bitmask_offset = current_bitmask_offset;
          w->level_ptr = current_level_ptr;
          w->output_ptr = current_output_ptr;

          // increment
          if(tag & 1){
            current_output_ptr += (length * 8 * type_size); 
            current_bitmask_offset += (length * 8);
            values_processed += (length * 8);
            if(page_bitmask_ptr){ 
              current_level_ptr += length;
            }
            /*
            if(page_idx == 1){
              printf("V0 (%d): %d\n", work_units_processed, length * 8);
            }
            */
          } else {
            current_output_ptr += (length * type_size);
            current_bitmask_offset += length;
            values_processed += length;
            if(page_bitmask_ptr){              
              current_level_ptr++;
            }
            /*
            if(page_idx == 1){
              printf("V1 (%d): %d\n", work_units_processed, length);
            }
            */
          }          
        }        
      }
    }
    __syncthreads();    

    continue;

    work_unit *w = &work[local_warp_id];    

    // step 2, generate the offsets into the input data for each work unit to read.
    if(w->length > 0){
      if(warp_lane == 0){        
        w->input_pos = 0;       
      }
      
      // rle encoded run. we will have to sift through all the values
      if(w->tag & 1){
        std::size_t num_rounds = utility::roundUpDiv(w->length * 8, 32);        
        for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
          std::size_t bit_idx         = round_idx * 32 + warp_lane;
          std::size_t byte_idx        = bit_idx / 8;
          std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

          uint8_t mask             = 0;
          cudf::size_type exclusive_output = 0; // output position per thread
          cudf::size_type warp_aggregate   = 0; // total # of values decoded for the whole warp

          if (byte_idx < w->length) {
            uint8_t current_byte = w->level_ptr[byte_idx];
            // mask will be either 0 or 1, indicating whether the current value is NULL
            mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
          }

          WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);
          if(warp_lane == 0){
            w->input_pos += warp_aggregate;
          }
        }
      } 
      // the simple case - just a repeated value (for non-nullable columns we always go through here)      
      else {
        if(warp_lane == 0){
          uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *w->level_ptr;
          w->input_pos += repeated_value ? w->length : 0;
        }
      }   
    }
    __syncthreads();    

    // generate offsets
    if(local_warp_id == 0){
      cudf::size_type w_pos = warp_lane < num_warps_per_block ? w[warp_lane].input_pos : 0;
      cudf::size_type warp_aggregate;
      WarpScan(temp_storage[0]).ExclusiveScan(w_pos, w_pos, cudf::size_type{0}, cub::Sum(), warp_aggregate);      
      if(warp_lane < num_warps_per_block){
        // absolute position
        w[warp_lane].input_pos = w_pos + ((current_data_ptr - page_data_cur) / type_size);
      }      
      current_data_ptr += (warp_aggregate * type_size);

      // if we have dictionary data, generate it now
      #if 0
      if(dict && warp_lane == 0){
        for(int idx=0; idx<num_warps_per_block; idx++){
          work_unit *w = &work[idx];
          if(w->length <= 0){
            break;
          }          
          /*
          if(w->input_pos < 256 && page_idx == 1){
            printf("WU %d (length %d)\n", idx, w->tag & 1 ? w->length * 8 : w->length);
          }          
          */
          dict_advance_to(&dict_cur, w->input_pos, page_idx);   
          w->dict.copy_pos(dict_cur);
        }
      }
      #endif
    }
    __syncthreads();

    /*   
    if(local_warp_id == 0 && warp_lane == 0 && page_idx == 83){
      for(int idx=0; idx<num_warps_per_block; idx++){
        if(w[idx].length > 0){
          uint64_t level_dist = w[idx].level_ptr - level_start;
          printf("WU(%d): %d, %d(%d), %d, %lu\n", wu_index, w[idx].length, (int)w[idx].tag, w[idx].tag & 1 ? 1 : 0, w[idx].input_pos, level_dist);

          wu_index++;
        }
      }
    }
    __syncthreads();
    */
       
    // process the work units. 1 warp == 1 work unit    
    if(w->length > 0){
      // process the work units
      if (w->tag & 1) {
        // bitpacked run
        // For a flat data type like integer, each value has 1 bit stored in the definition level: 0
        // for NULL, and 1 for not NULL. To copy values, note that Parquet does not store NULL
        // values, while Arrow does. So, we use a warp scan to calculate the input location. To copy
        // null masks, we can directly copy from the bitpacked definition levels.
        std::size_t num_rounds = utility::roundUpDiv(w->length * 8, 32);
        int input_pos = w->input_pos;
        for (std::size_t round_idx = 0; round_idx < num_rounds; round_idx++) {
          std::size_t bit_idx         = round_idx * 32 + warp_lane;
          std::size_t byte_idx        = bit_idx / 8;
          std::size_t bit_idx_in_byte = bit_idx - byte_idx * 8;

          uint8_t mask             = 0;
          cudf::size_type exclusive_output = 0; // output position per thread
          cudf::size_type warp_aggregate   = 0; // total # of values decoded for the whole warp

          if (byte_idx < w->length) {
            uint8_t current_byte = w->level_ptr[byte_idx];
            // mask will be either 0 or 1, indicating whether the current value is NULL
            mask = (current_byte & (1 << bit_idx_in_byte)) >> bit_idx_in_byte;
          }

          WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);
                    
          if(dict){
            // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane);
            dict_buffer_to(&w->dict, input_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
          }
                    
          if (mask) {            
            auto const dict_pos = (input_pos + exclusive_output) & (dict_buf_size - 1);
            auto const dict_val = w->dict.dict_idx[dict_pos];        
            auto const src = dict ? dict + (dict_val * type_size)
                                  : (input_data_ptr + (input_pos * type_size)) + (exclusive_output * type_size);
                                  
            // auto const src = (input_data_ptr + (input_pos * type_size)) + (exclusive_output * type_size);
            auto const output_pos = ((w->output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;
            /*
            if(page_idx == 1 && output_pos < 256 && local_warp_id == 1){
              printf("COPYA(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", local_warp_id, output_pos, dict_pos, dict_val);
            }
            */
            switch(type_size){
            case 4: copy_val4(w->output_ptr + (bit_idx * type_size), src); break;
            case 8: copy_val8(w->output_ptr + (bit_idx * type_size), src); break;
            default: break;
            }

            thread_valid_count++;
          }          

          input_pos += warp_aggregate;
        }
                
        if(page_bitmask_ptr != nullptr){
          copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, w->bitmask_offset, w->level_ptr, w->length, local_warp_id);
        }
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
        uint8_t repeated_value = page_bitmask_ptr == nullptr ? 1 : *w->level_ptr;

        if (repeated_value) {
          uint32_t output_base_idx = 0;
          int input_pos = w->input_pos;
          do {
            int warp_aggregate = output_base_idx + 32 > w->length ? w->length - output_base_idx : 32;
                        
            if(dict){
              dict_buffer_to(&w->dict, input_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
            }

            if(output_base_idx + warp_lane < w->length){    
              auto const dict_pos = (input_pos + warp_lane) & (dict_buf_size - 1);
              auto const dict_val = w->dict.dict_idx[dict_pos];
              auto const output_idx = output_base_idx + warp_lane;
              auto const src = dict ? dict + (dict_val * type_size)
                                    : (input_data_ptr + (input_pos * type_size)) + output_idx * type_size;

              auto const output_pos = ((w->output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;
              switch(type_size){              
              case 4: copy_val4(w->output_ptr + (output_idx * type_size), src); break;
              case 8: copy_val8(w->output_ptr + (output_idx * type_size), src); break;
              default: break;
              }

              thread_valid_count++;
            }            

            input_pos += warp_aggregate;
            output_base_idx += warp_aggregate;
          } while(output_base_idx < w->length);
                    
          if(page_bitmask_ptr != nullptr){
            set_validity_bits_safe(page_bitmask_ptr, w->bitmask_offset, w->length);
          }          
        }
      }    
    }       

    __syncthreads();
  }

  // compute validity count if applicable
  if(page_bitmask_ptr){    
    using BlockReduce = cub::BlockReduce<cudf::size_type, num_warps_per_block * 32>;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    cudf::size_type valid_count = BlockReduce(temp_storage).Sum(thread_valid_count);
    if(anchor_thread){
      output_null_counts[page_idx] = num_rows[page_idx] - valid_count;
    }
  }
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

  // const bool anchor_thread = local_warp_id == 0 && warp_lane == 0;

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
    auto const level_start = current_level_ptr;
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
        di->read_pos = 0;
        di->dict_batch_len = 0;
      }
      __syncwarp();
    }

    // pointer to the start of the values section 
    const uint8_t* input_data_ptr          = page_data_cur;
    const uint8_t* current_data_ptr        = input_data_ptr;
    int current_data_pos                   = 0;                       // by value index. tracks current_data_ptr
    uint8_t* current_output_ptr            = output_data[page_idx];   // starts at the first output row for the page    
    // cudf::size_type current_bitmask_offset = output_bitmask_offset[page_idx];
    cudf::size_type current_bitmask_offset = 0;
    cudf::size_type type_size = page_type_size[page_idx];
    cudf::size_type values_processed = 0;
    cudf::size_type valid_count = 0;    

    // int wu_index = 0;

    // Keep going until the definition levels have been completely parsed
    // Note that the end of the definition level section is the same as the start of the values
    // section
    int wu_index = 0;
    while ((reinterpret_cast<uintptr_t>(current_level_ptr) < reinterpret_cast<uintptr_t>(input_data_ptr)) ||
           (values_processed < num_rows[page_idx])) {

      //uint64_t level_dist = current_level_ptr - page_start;
      uint8_t tag     = page_bitmask_ptr == nullptr ? 0 : *current_level_ptr;
      uint32_t length = page_bitmask_ptr == nullptr ? num_rows[page_idx] : calculate_run_length(current_level_ptr);

      /*
      if(page_idx == 1 && warp_lane == 0 && values_processed < 256){
        printf("WU %d length(%d)\n", wu_index++, tag & 1 ? length * 8 : length);
      }
      */
      /*
      if(page_idx == 1 && warp_lane == 0 && values_processed < 1000){
        printf("V: %d, level_dist(%lu), tag(%d), rows(%d)\n", tag & 1 ? length * 8 : length, level_dist, (int)tag, num_rows[page_idx]);
      }
      */
      
      /*
      __syncthreads();
      if(anchor_thread && page_idx == 0){
        uint64_t level_dist = current_level_ptr - page_start;
        uint64_t dist = current_data_ptr - input_data_ptr;
        //printf("WU(%d): %d, %d, %lu, %lu\n", wu_index, length, (int)tag, dist, level_dist);
        wu_index++;
      }
      */

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

          WarpScan(temp_storage[local_warp_id]).ExclusiveSum(mask, exclusive_output, warp_aggregate);          
                    
          if(dict){
            dict_buffer_to2(di, current_data_pos + warp_aggregate, warp_lane);
            // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
          }          
          
          if (mask) {
            auto const dict_pos = dict ? (current_data_pos + exclusive_output) & (dict_buf_size - 1) : 0;
            auto const dict_val = dict ? di->dict_idx[dict_pos] : 0;
            auto const src = dict ? dict + (dict_val * type_size)
                                  : current_data_ptr + exclusive_output * type_size;
            
            /*
            uint64_t output_pos = ((current_output_ptr + (bit_idx * type_size)) - output_data[page_idx]) / type_size;
            if(page_idx == 1 && output_pos < 256){
              printf("COPYA0(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
            } 
            */           

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
          /*
          if(warp_lane == 0){
            auto real_offset = current_bitmask_offset - output_bitmask_offset[page_idx];
            if(real_offset + (length * 8) > ((num_rows[page_idx] + 7) / 8) * 8){
              printf("BAD: %d, %d, (%d) %d\n", real_offset, length * 8, real_offset + (length * 8), ((num_rows[page_idx] + 7) / 8) * 8);
            }
          }
          */
          valid_count += copy_validity_bits_safe<num_warps_per_block>(page_bitmask_ptr, current_bitmask_offset, output_bitmask_offset[page_idx], num_rows[page_idx], current_level_ptr, length, local_warp_id);
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
              dict_buffer_to2(di, current_data_pos + warp_aggregate, warp_lane);
              // dict_buffer_to(di, current_data_pos + warp_aggregate, warp_lane, page_idx, local_warp_id);
            }
            
            if(output_base_idx + warp_lane < length){              
              auto const output_idx = output_base_idx + warp_lane;

              auto const dict_pos = dict ? (current_data_pos + warp_lane) & (dict_buf_size - 1) : 0;
              auto const dict_val = dict ? di->dict_idx[dict_pos] : 0;
              auto const src = dict ? dict + (dict_val * type_size)
                                    : current_data_ptr + output_idx * type_size;                                                          
              
              /*
              uint64_t output_pos = ((current_output_ptr + (output_idx * type_size)) - output_data[page_idx]) / type_size;
              if(page_idx == 1 && output_pos < 256){                
                printf("COPYA1(%d): output_pos(%lu) <- dict_pos(%d, value:%d)\n", warp_lane, output_pos, dict_pos, dict_val);
              }
              */

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
            set_validity_bits_safe(page_bitmask_ptr, current_bitmask_offset + output_bitmask_offset[page_idx], length);
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

      //printf("\tUnsupported decimal type(%d)\n", (int)type);
      default: return {cudf::data_type{cudf::type_id::EMPTY}, 0};
    }
    break;
  // these will need downconversion handling
  case cudf::io::parquet::ConvertedType::UINT_8: 
  case cudf::io::parquet::ConvertedType::INT_8:     
  case cudf::io::parquet::ConvertedType::UINT_16: 
  case cudf::io::parquet::ConvertedType::INT_16:
    //printf("\tUnsupported int type(%d)\n", (int)converted_type);
    return {cudf::data_type{cudf::type_id::EMPTY}, 0};

  // timestamp stuff needs special handling as well
  case cudf::io::parquet::ConvertedType::DATE: 
  case cudf::io::parquet::ConvertedType::TIME_MILLIS: 
  case cudf::io::parquet::ConvertedType::TIME_MICROS:     
  case cudf::io::parquet::ConvertedType::TIMESTAMP_MILLIS: 
  case cudf::io::parquet::ConvertedType::TIMESTAMP_MICROS:
    //printf("\tUnsupported timestamp type(%d)\n", (int)converted_type);
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
  
  //printf("\tUnsupported other type(%d, %d)\n", (int)converted_type, (int)type);
  return {cudf::data_type{cudf::type_id::EMPTY}, 0};
}

std::optional<PageInfo> to_nvcomp(cudf::io::parquet::gpu::PageInfo const& page, cudf::io::parquet::gpu::ColumnChunkDesc const& chunk)
{   
  // reject lists.
  if(chunk.max_level[cudf::io::parquet::gpu::level_type::REPETITION] > 0){
    //printf("MAX REP: %d\n", chunk.max_level[cudf::io::parquet::gpu::level_type::REPETITION]);
    //printf("Skipping list page\n");
    return std::nullopt;
  }
  if(chunk.max_level[cudf::io::parquet::gpu::level_type::DEFINITION] > 1){
    //printf("MAX DEF: %d\n", chunk.max_level[cudf::io::parquet::gpu::level_type::DEFINITION]);
    // printf("VMAP(%d): %lu\n", (int)chunk.max_level[cudf::io::parquet::gpu::level_type::DEFINITION], (int64_t)chunk.valid_map_simple);
    //printf("Skipping list page (def level > 1)\n");
    return std::nullopt;
  }

  /*
  if(page.encoding != cudf::io::parquet::Encoding::PLAIN_DICTIONARY){
    return std::nullopt;
  }
  */

  auto const supported_encoding = page.encoding == cudf::io::parquet::Encoding::PLAIN || page.encoding == cudf::io::parquet::Encoding::PLAIN_DICTIONARY;
  auto const type = to_type(static_cast<cudf::io::parquet::Type>(chunk.data_type & 0x7), 
                            static_cast<cudf::io::parquet::ConvertedType>(chunk.converted_type), 
                            chunk.decimal_scale);  
  if(!supported_encoding || type.first.id() == cudf::type_id::EMPTY){        
    /*
    if(!supported_encoding){
      printf("Skipping unsupported page encoding : %d\n", static_cast<int>(page.encoding));
    }
    if(type.first.id() == cudf::type_id::EMPTY){
      printf("Skipping unsupported cudf type : %d\n", static_cast<int>(type.first.id()));
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
           chunk.src_col_index,
           chunk.dict_page_index});
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

  // printf("NVCOMP processing: %lu pages\n", pages.size());

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
  std::vector<cudf::size_type> dict_page_indices(num_pages);

  for (std::size_t page_idx = 0; page_idx < num_pages; page_idx++) {
    decompressed_ptrs[page_idx]        = pages[page_idx].page_data;   // will be null for dictionary pages
    uncompressed_bytes[page_idx]       = pages[page_idx].uncompressed_bytes;
    output_data_ptrs[page_idx]         = static_cast<uint8_t*>(pages[page_idx].output_data_ptr);    
    /*
    if(output_data_ptrs[page_idx] != nullptr){
      cudaMemset(output_data_ptrs[page_idx], 0xff, output_type_sizes[page_idx] * num_rows[num_pages]);
    }
    */
    output_null_mask_ptrs[page_idx]    = pages[page_idx].output_null_mask_ptr;
    output_null_mask_offsets[page_idx] = pages[page_idx].output_null_mask_offset;
    output_type_sizes[page_idx] = pages[page_idx].type_size;
    dicts[page_idx] = pages[page_idx].dict;
    num_rows[page_idx] = pages[page_idx].num_rows;
    output_src_col_indices[page_idx] = pages[page_idx].src_col_index;
    dict_page_indices[page_idx] = pages[page_idx].dict_page_index;
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
  rmm::device_vector<cudf::size_type> device_dict_page_indices = dict_page_indices;  
  
  if(Use_nvcomp_decode_path2){      
    // constexpr int num_warps_per_block = 24;
    constexpr int num_warps_per_block = 3;  
    decode_pages_kernel4<num_warps_per_block>
      <<<num_pages, num_warps_per_block * 32, 0, stream.value()>>>(
        thrust::raw_pointer_cast(device_decompressed_ptrs.data()),
        thrust::raw_pointer_cast(device_uncompressed_bytes.data()),
        thrust::raw_pointer_cast(device_output_data_ptrs.data()),
        thrust::raw_pointer_cast(device_output_null_mask_ptrs.data()),
        thrust::raw_pointer_cast(device_output_null_mask_offsets.data()),
        thrust::raw_pointer_cast(device_output_type_sizes.data()),
        thrust::raw_pointer_cast(device_dicts.data()),
        thrust::raw_pointer_cast(device_num_rows.data()),
        device_output_null_counts.device_ptr(),
        thrust::raw_pointer_cast(device_dict_page_indices.data()),
        num_pages);
  } else {
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
  }

  return {std::move(output_src_col_indices), std::move(device_output_null_counts)};
}

// given an input list of cuIO parquet reader pages, dispatch as many of them as we can using the nvcomp
// reader. return a vector of pages that could not be processed.
std::tuple<hostdevice_vector<cudf::io::parquet::gpu::PageInfo>,
           std::vector<cudf::size_type>,
           hostdevice_vector<cudf::size_type>>                                  
decode_relevant_pages(hostdevice_vector<cudf::io::parquet::gpu::ColumnChunkDesc>& chunks,
                      hostdevice_vector<cudf::io::parquet::gpu::PageInfo>& _pages,
                      rmm::cuda_stream_view stream,
                      rmm::cuda_stream_view return_stream)
{  
  cudf::size_type dict_pages = 0;

  hostdevice_vector<cudf::io::parquet::gpu::PageInfo> remainder(0, _pages.size(), stream);
  std::vector<PageInfo> nvcomp_pages;
  nvcomp_pages.reserve(_pages.size());
  std::for_each(_pages.begin(), _pages.end(), [&](cudf::io::parquet::gpu::PageInfo const &p){
    auto nvcomp = to_nvcomp(p, chunks[p.chunk_idx]);
    if(nvcomp.has_value()){
      nvcomp_pages.push_back(nvcomp.value());
      if(nvcomp_pages.back().dict != nullptr){
        dict_pages++;
      }
    } else {
      remainder.insert(p);
    }
  });

  // printf("NVC: %lu total pages (%d dict pages). nvcomp processing(%lu), cuIO processing(%lu)\n", _pages.size(), dict_pages, nvcomp_pages.size(), remainder.size());

  // invoke nvcomp decoding
  auto [src_col_indices, null_counts] = decode_values(nvcomp_pages, stream);
  
  // return the remainder of the pages to cuIO for decoding.
  remainder.host_to_device(return_stream);
  return {std::move(remainder), std::move(src_col_indices), std::move(null_counts)};
}

} // namespace parquet

} // namespace experimental
