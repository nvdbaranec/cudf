/*
 * Copyright (c) 2018, NVIDIA CORPORATION.
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

#include <rmm/thrust_rmm_allocator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <cudf/detail/utilities/release_assert.cuh>
#include <cudf/utilities/bit.hpp>
#include <io/utilities/block_utils.cuh>

#include "parquet_gpu.h"
#include "printer.hpp"

#define LOG2_NTHREADS (5 + 2)
#define NTHREADS (1 << LOG2_NTHREADS)
#define NZ_BFRSZ (NTHREADS * 2)

inline __device__ uint32_t rotl32(uint32_t x, uint32_t r)
{
  return __funnelshift_l(x, x, r);  // (x << r) | (x >> (32 - r));
};

inline __device__ int rolling_index(int index) { return index & (NZ_BFRSZ - 1); }

constexpr int column_watch = 4;

namespace cudf {
namespace io {
namespace parquet {
namespace gpu {

struct page_state_s {
  const uint8_t *lvl_start[2];  // [def,rep]
  const uint8_t *data_start;
  const uint8_t *data_end;
  const uint8_t *dict_base;   // ptr to dictionary page data
  int32_t dict_size;          // size of dictionary data

  int32_t first_row;          // First row in page to output
  int32_t num_rows;           // Rows in page to decode (including rows to be skipped)
  
  int32_t first_output_value; // First value in page to output
  int32_t num_values;         // total # of values in the page

  int32_t dtype_len;          // Output data type length
  int32_t dtype_len_in;         // Can be larger than dtype_len if truncating 32-bit into 8-bit
  int32_t dict_bits;             // # of bits to store dictionary indices
  uint32_t dict_run;
  int32_t dict_val;
  uint32_t initial_rle_run[2];   // [def,rep]
  int32_t initial_rle_value[2];  // [def,rep]
  int32_t error;
  PageInfo page;
  ColumnChunkDesc col;

  // only used in the preprocess step
  //uint32_t **valid_map;
  //int32_t *valid_map_offset;

  // value decoding
  //uint8_t **data_out;
  int32_t nz_count;  // number of valid entries in nz_idx (write position in circular buffer)
  int32_t dict_pos;  // write position of dictionary indices
  int32_t out_pos;   // read position of final output
  int32_t ts_scale;  // timestamp scale: <0: divide by -ts_scale, >0: multiply by ts_scale
  uint32_t nz_idx[NZ_BFRSZ];    // circular buffer of non-null value positions
  uint32_t dict_idx[NZ_BFRSZ];  // Dictionary index, boolean, or string offset values
  uint32_t str_len[NZ_BFRSZ];   // String length for plain encoding of strings

  // level decoding
  int page_value_count;
  uint32_t rep[NZ_BFRSZ];
  uint32_t def[NZ_BFRSZ];
  int32_t lvl_count[2];
  int32_t rep_count;
  int32_t def_count;  
};

/**
 * @brief Computes a 32-bit hash when given a byte stream and range.
 *
 * MurmurHash3_32 implementation from
 * https://github.com/aappleby/smhasher/blob/master/src/MurmurHash3.cpp
 *
 * MurmurHash3 was written by Austin Appleby, and is placed in the public
 * domain. The author hereby disclaims copyright to this source code.
 *
 * @param[in] key The input data to hash
 * @param[in] len The length of the input data
 * @param[in] seed An initialization value
 *
 * @return The hash value
 */
__device__ uint32_t device_str2hash32(const char *key, size_t len, uint32_t seed = 33)
{
  const uint8_t *p  = reinterpret_cast<const uint8_t *>(key);
  uint32_t h1       = seed, k1;
  const uint32_t c1 = 0xcc9e2d51;
  const uint32_t c2 = 0x1b873593;
  int l             = len;
  // body
  while (l >= 4) {
    k1 = p[0] | (p[1] << 8) | (p[2] << 16) | (p[3] << 24);
    k1 *= c1;
    k1 = rotl32(k1, 15);
    k1 *= c2;
    h1 ^= k1;
    h1 = rotl32(h1, 13);
    h1 = h1 * 5 + 0xe6546b64;
    p += 4;
    l -= 4;
  }
  // tail
  k1 = 0;
  switch (l) {
    case 3: k1 ^= p[2] << 16;
    case 2: k1 ^= p[1] << 8;
    case 1:
      k1 ^= p[0];
      k1 *= c1;
      k1 = rotl32(k1, 15);
      k1 *= c2;
      h1 ^= k1;
  }
  // finalization
  h1 ^= len;
  h1 ^= h1 >> 16;
  h1 *= 0x85ebca6b;
  h1 ^= h1 >> 13;
  h1 *= 0xc2b2ae35;
  h1 ^= h1 >> 16;
  return h1;
}

/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(const uint8_t *&cur, const uint8_t *end)
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
 * @brief Parse the beginning of the level section (definition or repetition),
 * initializes the initial RLE run & value, and returns the section length
 *
 * @param[in,out] s The page state
 * @param[in] cur The current data position
 * @param[in] end The end of the data
 * @param[in] level_bits The bits required

 */
__device__ uint32_t InitLevelSection(page_state_s *s,
                                     const uint8_t *cur,
                                     const uint8_t *end,
                                     level_type lvl)
{
  int32_t len;
  int level_bits = s->col.level_bits[lvl];
  int encoding   = lvl == level_type::DEFINITION ? s->page.definition_level_encoding
                                               : s->page.repetition_level_encoding;

  if (level_bits == 0) {
    len                       = 0;
    s->initial_rle_run[lvl]   = s->page.num_values * 2;  // repeated value
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
  } else if (encoding == RLE) {
    if (cur + 4 < end) {
      uint32_t run;
      len = 4 + (cur[0]) + (cur[1] << 8) + (cur[2] << 16) + (cur[3] << 24);
      cur += 4;
      run                     = get_vlq32(cur, end);
      s->initial_rle_run[lvl] = run;
      if (!(run & 1)) {
        int v = (cur < end) ? cur[0] : 0;
        cur++;
        if (level_bits > 8) {
          v |= ((cur < end) ? cur[0] : 0) << 8;
          cur++;
        }
        s->initial_rle_value[lvl] = v;
      }
      s->lvl_start[lvl] = cur;
      if (cur > end) { s->error = 2; }
    } else {
      len      = 0;
      s->error = 2;
    }
  } else if (encoding == BIT_PACKED) {
    len                       = (s->page.num_values * level_bits + 7) >> 3;
    s->initial_rle_run[lvl]   = ((s->page.num_values + 7) >> 3) * 2 + 1;  // literal run
    s->initial_rle_value[lvl] = 0;
    s->lvl_start[lvl]         = cur;
  } else {
    s->error = 3;
    len      = 0;
  }
  return (uint32_t)len;
}

/**
 * @brief Decode definition or repetition levels
 *
 * @param[in,out] s Page state input/output
 * @param[in] t target_count Target count of non-NULL values on output
 * @param[in] t Warp0 thread ID (0..31)
 * @param[in] lvl The level type we are decoding - DEFINITION or REPETITION
 */
__device__ int32_t
gpuDecodeStream(uint32_t *output, page_state_s *s, int32_t count, int t, level_type lvl)
{
  const uint8_t *cur_def    = s->lvl_start[lvl];
  const uint8_t *end        = s->data_start;
  uint32_t level_run        = s->initial_rle_run[lvl];
  int32_t level_val         = s->initial_rle_value[lvl];
  int level_bits            = s->col.level_bits[lvl];
  int max_level             = s->col.max_level[lvl];
  int32_t num_values        = s->num_values;
  int32_t value_count       = s->lvl_count[lvl];
  int32_t batch_coded_count = 0;

  if (!t && s->col.col_index == column_watch) {
        //printf("GDS START %d, (%d) : decoding %d values, current count %d, %d\n", t, lvl, count,
        //value_count, max_level);
  }

  // while (batch_coded_count < count && value_count < num_values) {
  while (value_count < count && value_count < num_values) {
    int batch_len, is_valid;
    uint32_t valid_mask;
    if (level_run <= 1) {
      // Get a new run symbol from the byte stream
      int sym_len = 0;
      if (!t) {
        const uint8_t *cur = cur_def;
        if (cur < end) { level_run = get_vlq32(cur, end); }
        if (!(level_run & 1)) {
          if (cur < end) level_val = cur[0];
          cur++;
          if (level_bits > 8) {
            if (cur < end) level_val |= cur[0] << 8;
            cur++;
          }
        }
        if (cur > end || level_run <= 1) { s->error = 0x10; }
        sym_len = (int32_t)(cur - cur_def);
        __threadfence_block();
      }
      sym_len   = SHFL0(sym_len);
      level_val = SHFL0(level_val);
      level_run = SHFL0(level_run);
      cur_def += sym_len;
    }
    if (s->error) { break; }

    batch_len = min(num_values - value_count, 32);
    if (level_run & 1) {
      // Literal run
      int batch_len8;
      batch_len  = min(batch_len, (level_run >> 1) * 8);
      batch_len8 = (batch_len + 7) >> 3;
      if (t < batch_len) {
        int bitpos         = t * level_bits;
        const uint8_t *cur = cur_def + (bitpos >> 3);
        bitpos &= 7;
        if (cur < end) level_val = cur[0];
        cur++;
        if (level_bits > 8 - bitpos && cur < end) {
          level_val |= cur[0] << 8;
          cur++;
          if (level_bits > 16 - bitpos && cur < end) level_val |= cur[0] << 16;
        }
        level_val = (level_val >> bitpos) & ((1 << level_bits) - 1);
      }
      level_run -= batch_len8 * 2;
      cur_def += batch_len8 * level_bits;
    } else {
      // Repeated value
      batch_len = min(batch_len, level_run >> 1);
      level_run -= batch_len * 2;
    }
    if (t < batch_len) {
      int idx = value_count + t;
      output[idx & (NZ_BFRSZ - 1)] = level_val;
    }
    batch_coded_count += batch_len;
    value_count += batch_len;
  }

  // update the stream info
  __syncthreads();
  if (!t) {
    s->lvl_start[lvl]         = cur_def;
    s->initial_rle_run[lvl]   = level_run;
    s->initial_rle_value[lvl] = level_val;
    s->lvl_count[lvl]         = value_count;

    if (!t && s->col.col_index == column_watch) {
       //printf("GDS END (%d) : decoded %d values, final count %d\n", lvl, batch_coded_count,
       //value_count);
    }
  }

  // return how many values we actually processed
  return batch_coded_count;
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
__device__ int gpuDecodeDictionaryIndices(volatile page_state_s *s, int target_pos, int t)
{
  const uint8_t *end = s->data_end;
  int dict_bits      = s->dict_bits;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      const uint8_t *cur = s->data_start;
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
                if (bytecnt > 3) { run_val |= cur[3] << 24; }
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
    SYNCWARP();
    is_literal = SHFL0(is_literal);
    batch_len  = SHFL0(batch_len);
    if (t < batch_len) {
      int dict_idx = s->dict_val;
      if (is_literal) {
        int32_t ofs      = (t - ((batch_len + 7) & ~7)) * dict_bits;
        const uint8_t *p = s->data_start + (ofs >> 3);
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
      s->dict_idx[(pos + t) & (NZ_BFRSZ - 1)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Performs RLE decoding of dictionary indexes, for when dict_size=1
 *
 * @param[in,out] s Page state input/output
 * @param[in] target_pos Target write position
 * @param[in] t Thread ID
 *
 * @return The new output position
 */
__device__ int gpuDecodeRleBooleans(volatile page_state_s *s, int target_pos, int t)
{
  const uint8_t *end = s->data_end;
  int pos            = s->dict_pos;

  while (pos < target_pos) {
    int is_literal, batch_len;
    if (!t) {
      uint32_t run       = s->dict_run;
      const uint8_t *cur = s->data_start;
      if (run <= 1) {
        run = (cur < end) ? get_vlq32(cur, end) : 0;
        if (!(run & 1)) {
          // Repeated value
          s->dict_val = (cur < end) ? cur[0] & 1 : 0;
          cur++;
        }
      }
      if (run & 1) {
        // Literal batch: must output a multiple of 8, except for the last batch
        int batch_len_div8;
        batch_len = max(min(32, (int)(run >> 1) * 8), 1);
        if (batch_len >= 8) { batch_len &= ~7; }
        batch_len_div8 = (batch_len + 7) >> 3;
        run -= batch_len_div8 * 2;
        cur += batch_len_div8;
      } else {
        batch_len = max(min(32, (int)(run >> 1)), 1);
        run -= batch_len * 2;
      }
      s->dict_run   = run;
      s->data_start = cur;
      is_literal    = run & 1;
      __threadfence_block();
    }
    SYNCWARP();
    is_literal = SHFL0(is_literal);
    batch_len  = SHFL0(batch_len);
    if (t < batch_len) {
      int dict_idx;
      if (is_literal) {
        int32_t ofs      = t - ((batch_len + 7) & ~7);
        const uint8_t *p = s->data_start + (ofs >> 3);
        dict_idx         = (p < end) ? (p[0] >> (ofs & 7u)) & 1 : 0;
      } else {
        dict_idx = s->dict_val;
      }
      s->dict_idx[(pos + t) & (NZ_BFRSZ - 1)] = dict_idx;
    }
    pos += batch_len;
  }
  return pos;
}

/**
 * @brief Parses the length and position of strings
 *
 * @param[in,out] s Page state input/output
 * @param[in] target_pos Target output position
 * @param[in] t Thread ID
 *
 * @return The new output position
 */
__device__ void gpuInitStringDescriptors(volatile page_state_s *s, int target_pos, int t)
{
  int pos = s->dict_pos;
  // This step is purely serial
  if (!t) {
    const uint8_t *cur = s->data_start;
    int dict_size      = s->dict_size;
    int k              = s->dict_val;

    while (pos < target_pos) {
      int len;
      if (k + 4 <= dict_size) {
        len = (cur[k]) | (cur[k + 1] << 8) | (cur[k + 2] << 16) | (cur[k + 3] << 24);
        k += 4;
        if (k + len > dict_size) { len = 0; }
      } else {
        len = 0;
      }
      s->dict_idx[pos & (NZ_BFRSZ - 1)] = k;
      s->str_len[pos & (NZ_BFRSZ - 1)]  = len;
      k += len;
      pos++;
    }
    s->dict_val = k;
    __threadfence_block();
  }
}

/**
 * @brief Output a string descriptor
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dstv Pointer to row output data (string descriptor or 32-bit hash)
 */
inline __device__ void gpuOutputString(volatile page_state_s *s,
                                       int src_pos,
                                       void *dstv,
                                       int value_idx)
{
  const char *ptr = NULL;
  size_t len      = 0;

  if (s->dict_base) {
    // String dictionary
    uint32_t dict_pos =
      (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] * sizeof(nvstrdesc_s) : 0;
    if (dict_pos < (uint32_t)s->dict_size) {
      const nvstrdesc_s *src = reinterpret_cast<const nvstrdesc_s *>(s->dict_base + dict_pos);
      ptr                    = src->ptr;
      len                    = src->count;
    }
  } else {
    // Plain encoding
    uint32_t dict_pos = s->dict_idx[src_pos & (NZ_BFRSZ - 1)];
    if (dict_pos <= (uint32_t)s->dict_size) {
      ptr = reinterpret_cast<const char *>(s->data_start + dict_pos);
      len = s->str_len[src_pos & (NZ_BFRSZ - 1)];
    }
  }
  if (s->dtype_len == 4) {
    // Output hash
    *reinterpret_cast<uint32_t *>(dstv) = device_str2hash32(ptr, len);
  } else {
    // Output string descriptor
    nvstrdesc_s *dst = reinterpret_cast<nvstrdesc_s *>(dstv);
    dst->ptr         = ptr;
    dst->count       = len;
  }
}

/**
 * @brief Output a boolean
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputBoolean(volatile page_state_s *s, int src_pos, uint8_t *dst)
{
  *dst = s->dict_idx[src_pos & (NZ_BFRSZ - 1)];
}

/**
 * @brief Store a 32-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint32_t *dst,
                                      const uint8_t *src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  // uint32_t tmp;
  // dst = &tmp;

  uint32_t bytebuf;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    bytebuf = *(const uint32_t *)(src8 + dict_pos);
    if (ofs) {
      uint32_t bytebufnext = *(const uint32_t *)(src8 + dict_pos + 4);
      bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
    }
  } else {
    bytebuf = 0;
  }
  *dst = bytebuf;
}

/**
 * @brief Store a 64-bit data element
 *
 * @param[out] dst ptr to output
 * @param[in] src8 raw input bytes
 * @param[in] dict_pos byte position in dictionary
 * @param[in] dict_size size of dictionary
 */
inline __device__ void gpuStoreOutput(uint2 *dst,
                                      const uint8_t *src8,
                                      uint32_t dict_pos,
                                      uint32_t dict_size)
{
  uint2 v;
  unsigned int ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos < dict_size) {
    v.x = *(const uint32_t *)(src8 + dict_pos + 0);
    v.y = *(const uint32_t *)(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *(const uint32_t *)(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
  } else {
    v.x = v.y = 0;
  }
  *dst = v;
}

/**
 * @brief Convert an INT96 Spark timestamp to 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputInt96Timestamp(volatile page_state_s *s, int src_pos, int64_t *dst)
{
  const uint8_t *src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    src8     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos + 4 < dict_size) {
    uint3 v;
    int64_t nanos, secs, days;
    v.x = *(const uint32_t *)(src8 + dict_pos + 0);
    v.y = *(const uint32_t *)(src8 + dict_pos + 4);
    v.z = *(const uint32_t *)(src8 + dict_pos + 8);
    if (ofs) {
      uint32_t next = *(const uint32_t *)(src8 + dict_pos + 12);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, v.z, ofs);
      v.z           = __funnelshift_r(v.z, next, ofs);
    }
    nanos = v.y;
    nanos <<= 32;
    nanos |= v.x;
    // Convert from Julian day at noon to UTC seconds
    days = static_cast<int32_t>(v.z);
    secs = (days - 2440588) *
           (24 * 60 * 60);  // TBD: Should be noon instead of midnight, but this matches pyarrow
    if (s->col.ts_clock_rate)
      ts = (secs * s->col.ts_clock_rate) +
           nanos / (1000000000 / s->col.ts_clock_rate);  // Output to desired clock rate
    else
      ts = (secs * 1000000000) + nanos;
  } else {
    ts = 0;
  }
  *dst = ts;
}

/**
 * @brief Output a 64-bit timestamp
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
inline __device__ void gpuOutputInt64Timestamp(volatile page_state_s *s, int src_pos, int64_t *dst)
{
  const uint8_t *src8;
  uint32_t dict_pos, dict_size = s->dict_size, ofs;
  int64_t ts;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    src8     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    src8     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  ofs = 3 & reinterpret_cast<size_t>(src8);
  src8 -= ofs;  // align to 32-bit boundary
  ofs <<= 3;    // bytes -> bits
  if (dict_pos + 4 < dict_size) {
    uint2 v;
    int64_t val;
    int32_t ts_scale;
    v.x = *(const uint32_t *)(src8 + dict_pos + 0);
    v.y = *(const uint32_t *)(src8 + dict_pos + 4);
    if (ofs) {
      uint32_t next = *(const uint32_t *)(src8 + dict_pos + 8);
      v.x           = __funnelshift_r(v.x, v.y, ofs);
      v.y           = __funnelshift_r(v.y, next, ofs);
    }
    val = v.y;
    val <<= 32;
    val |= v.x;
    // Output to desired clock rate
    ts_scale = s->ts_scale;
    if (ts_scale < 0) {
      // round towards negative infinity
      int sign = (val < 0);
      ts       = ((val + sign) / -ts_scale) + sign;
    } else {
      ts = val * ts_scale;
    }
  } else {
    ts = 0;
  }
  *dst = ts;
}

/**
 * @brief Powers of 10
 */
static const __device__ __constant__ double kPow10[40] = {
  1.0,   1.e1,  1.e2,  1.e3,  1.e4,  1.e5,  1.e6,  1.e7,  1.e8,  1.e9,  1.e10, 1.e11, 1.e12, 1.e13,
  1.e14, 1.e15, 1.e16, 1.e17, 1.e18, 1.e19, 1.e20, 1.e21, 1.e22, 1.e23, 1.e24, 1.e25, 1.e26, 1.e27,
  1.e28, 1.e29, 1.e30, 1.e31, 1.e32, 1.e33, 1.e34, 1.e35, 1.e36, 1.e37, 1.e38, 1.e39,
};

/**
 * @brief Output a decimal type ([INT32..INT128] + scale) as a 64-bit float
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 * @param[in] dtype Stored data type
 */
inline __device__ void gpuOutputDecimal(volatile page_state_s *s,
                                        int src_pos,
                                        double *dst,
                                        int dtype)
{
  const uint8_t *dict;
  uint32_t dict_pos, dict_size = s->dict_size, dtype_len_in;
  int64_t i128_hi, i128_lo;
  int32_t scale;
  double d;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    dict     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dtype_len_in = s->dtype_len_in;
  dict_pos *= dtype_len_in;
  // FIXME: Not very efficient (currently reading 1 byte at a time) -> need a variable-length
  // unaligned load utility function (both little-endian and big-endian versions)
  if (dtype == INT32) {
    int32_t lo32 = 0;
    for (unsigned int i = 0; i < dtype_len_in; i++) {
      uint32_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
      lo32 |= v << (i * 8);
    }
    i128_lo = lo32;
    i128_hi = lo32 >> 31;
  } else if (dtype == INT64) {
    int64_t lo64 = 0;
    for (unsigned int i = 0; i < dtype_len_in; i++) {
      uint64_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
      lo64 |= v << (i * 8);
    }
    i128_lo = lo64;
    i128_hi = lo64 >> 63;
  } else  // if (dtype == FIXED_LENGTH_BYTE_ARRAY)
  {
    i128_lo = 0;
    for (unsigned int i = dtype_len_in - min(dtype_len_in, 8); i < dtype_len_in; i++) {
      uint32_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
      i128_lo    = (i128_lo << 8) | v;
    }
    if (dtype_len_in > 8) {
      i128_hi = 0;
      for (unsigned int i = dtype_len_in - min(dtype_len_in, 16); i < dtype_len_in - 8; i++) {
        uint32_t v = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
        i128_hi    = (i128_hi << 8) | v;
      }
      if (dtype_len_in < 16) {
        i128_hi <<= 64 - (dtype_len_in - 8) * 8;
        i128_hi >>= 64 - (dtype_len_in - 8) * 8;
      }
    } else {
      if (dtype_len_in < 8) {
        i128_lo <<= 64 - dtype_len_in * 8;
        i128_lo >>= 64 - dtype_len_in * 8;
      }
      i128_hi = i128_lo >> 63;
    }
  }
  scale = s->col.decimal_scale;
  d     = Int128ToDouble_rn(i128_lo, i128_hi);
  *dst  = (scale < 0) ? (d * kPow10[min(-scale, 39)]) : (d / kPow10[min(scale, 39)]);
}

/**
 * @brief Output a small fixed-length value
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst Pointer to row output data
 */
template <typename T>
inline __device__ void gpuOutputFast(volatile page_state_s *s, int src_pos, T *dst)
{
  const uint8_t *dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    dict     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  gpuStoreOutput(dst, dict, dict_pos, dict_size);
}

/**
 * @brief Output a N-byte value
 *
 * @param[in,out] s Page state input/output
 * @param[in] src_pos Source position
 * @param[in] dst8 Pointer to row output data
 * @param[in] len Length of element
 */
static __device__ void gpuOutputGeneric(volatile page_state_s *s,
                                        int src_pos,
                                        uint8_t *dst8,
                                        int len)
{
  const uint8_t *dict;
  uint32_t dict_pos, dict_size = s->dict_size;

  if (s->dict_base) {
    // Dictionary
    dict_pos = (s->dict_bits > 0) ? s->dict_idx[src_pos & (NZ_BFRSZ - 1)] : 0;
    dict     = s->dict_base;
  } else {
    // Plain
    dict_pos = src_pos;
    dict     = s->data_start;
  }
  dict_pos *= (uint32_t)s->dtype_len_in;
  if (len & 3) {
    // Generic slow path
    for (unsigned int i = 0; i < len; i++) {
      dst8[i] = (dict_pos + i < dict_size) ? dict[dict_pos + i] : 0;
    }
  } else {
    // Copy 4 bytes at a time
    const uint8_t *src8 = dict;
    unsigned int ofs    = 3 & reinterpret_cast<size_t>(src8);
    src8 -= ofs;  // align to 32-bit boundary
    ofs <<= 3;    // bytes -> bits
    for (unsigned int i = 0; i < len; i += 4) {
      uint32_t bytebuf;
      if (dict_pos < dict_size) {
        bytebuf = *(const uint32_t *)(src8 + dict_pos);
        if (ofs) {
          uint32_t bytebufnext = *(const uint32_t *)(src8 + dict_pos + 4);
          bytebuf              = __funnelshift_r(bytebuf, bytebufnext, ofs);
        }
      } else {
        bytebuf = 0;
      }
      dict_pos += 4;
      *(uint32_t *)(dst8 + i) = bytebuf;
    }
  }
}

static __device__ bool setupLocalPageInfo(page_state_s *const s,
                                          PageInfo *pages,
                                          ColumnChunkDesc *chunks,
                                          size_t min_row,
                                          size_t num_rows,
                                          int32_t num_chunks)
{
  int page_idx = blockIdx.x;
  int t        = threadIdx.x;
  int chunk_idx;

  // Fetch page info
  // NOTE: Assumes that sizeof(PageInfo) <= 256 (and is padded to 4 bytes)
  if (t < sizeof(PageInfo) / sizeof(uint32_t)) {
    ((uint32_t *)&s->page)[t] = ((const uint32_t *)&pages[page_idx])[t];
  }
  __syncthreads();  
  if (s->page.flags & PAGEINFO_FLAGS_DICTIONARY) { return false; } 
  // Fetch column chunk info
  chunk_idx = s->page.chunk_idx;
  if ((uint32_t)chunk_idx < (uint32_t)num_chunks) {
    // NOTE: Assumes that sizeof(ColumnChunkDesc) <= 256 (and is padded to 4 bytes)
    if (t < sizeof(ColumnChunkDesc) / sizeof(uint32_t)) {
      ((uint32_t *)&s->col)[t] = ((const uint32_t *)&chunks[chunk_idx])[t];
    }
  }

  if (!t && s->col.col_index == column_watch) {
    /*
    printf(
      "PAGE SETUP START : page(%d), chunk_idx(%d/%d), col(%d), num_rows(%d), num_values(%d), "
      "chunk_row(%d), start_row(%d)\n",
      page_idx,
      chunk_idx,
      num_chunks,
      s->col.col_index,
      s->page.num_rows,
      s->page.num_values,
      s->page.chunk_row,
      s->col.start_row);
      */
  }

  // zero nested value and valid counts
  if(!t){
    printf("MND : %d\n", s->page.max_nesting_depth);
  }
  if (t < s->page.max_nesting_depth) {    
    s->page.nesting[t].o_value_count = 0;        
    s->page.nesting[t].o_valid_count = 0;
  }
  __syncthreads();

  if (!t) {
    s->error = 0;
    if (s->page.num_values > 0 && s->page.num_rows > 0) {
      uint8_t *cur          = s->page.page_data;
      uint8_t *end          = cur + s->page.uncompressed_page_size;      

      uint32_t dtype_len_out = s->col.data_type >> 3;
      s->ts_scale            = 0;
      // Validate data type
      switch (s->col.data_type & 7) {
        case BOOLEAN:
          s->dtype_len = 1;  // Boolean are stored as 1 byte on the output
          break;
        case INT32:
        case FLOAT: s->dtype_len = 4; break;
        case INT64:
          if (s->col.ts_clock_rate) {
            int32_t units = 0;
            if (s->col.converted_type == TIME_MICROS || s->col.converted_type == TIMESTAMP_MICROS)
              units = 1000000;
            else if (s->col.converted_type == TIME_MILLIS ||
                     s->col.converted_type == TIMESTAMP_MILLIS)
              units = 1000;
            if (units && units != s->col.ts_clock_rate)
              s->ts_scale = (s->col.ts_clock_rate < units) ? -(units / s->col.ts_clock_rate)
                                                           : (s->col.ts_clock_rate / units);
          }
          // Fall through to DOUBLE
        case DOUBLE: s->dtype_len = 8; break;
        case INT96: s->dtype_len = 12; break;
        case BYTE_ARRAY: s->dtype_len = sizeof(nvstrdesc_s); break;
        default:  // FIXED_LEN_BYTE_ARRAY:
          s->dtype_len = dtype_len_out;
          s->error |= (s->dtype_len <= 0);
          break;
      }
      // Special check for downconversions
      s->dtype_len_in = s->dtype_len;
      if (s->col.converted_type == DECIMAL) {
        s->dtype_len = 8;  // Convert DECIMAL to 64-bit float
      } else if ((s->col.data_type & 7) == INT32) {
        if (dtype_len_out == 1) s->dtype_len = 1;  // INT8 output
        if (dtype_len_out == 2) s->dtype_len = 2;  // INT16 output
      } else if ((s->col.data_type & 7) == BYTE_ARRAY && dtype_len_out == 4) {
        s->dtype_len = 4;  // HASH32 output
      } else if ((s->col.data_type & 7) == INT96) {
        s->dtype_len = 8;  // Convert to 64-bit timestamp
      }

      // our starting row (absolute index) is
      // col.start_row == absolute row index
      // page.chunk-row == relative row index within the chunk
      size_t page_start_row = s->col.start_row + s->page.chunk_row;
      // printf("PSR : %lu %d %lu\n", s->col.start_row, s->page.chunk_row, page_start_row);

      // during the decoding step we need to offset the global output buffers
      // for each level of nesting so that we write to the section this page
      // is reponsible for.  
      // - for flat schemas, we can do this directly by using row counts
      // - for nested schemas, these offsets are computed during the preprocess step
      if (s->col.column_data_base != nullptr) {                
        int max_depth   = s->col.max_level[level_type::REPETITION];
        for (int idx = 0; idx <= max_depth; idx++) {          
          int output_offset = max_depth == 0 ? page_start_row : s->page.nesting[idx].o_page_start_value;
          if(s->col.col_index == column_watch){
            //printf("PSV : %d\n", output_offset);
          }
          s->page.nesting[idx].data_out         = reinterpret_cast<uint8_t *>(s->col.column_data_base[idx]) + (output_offset * s->dtype_len);          
          s->page.nesting[idx].valid_map        = s->col.valid_map_base[idx];
          if(s->page.nesting[idx].valid_map != nullptr){
            s->page.nesting[idx].valid_map        += output_offset >> 5;
            s->page.nesting[idx].o_valid_map_offset = (int32_t)(output_offset & 0x1f);
          }          

          //printf("PTR : %lu, %lu, %d\n", (uint64_t)s->page.nesting[idx].data_out, (uint64_t)s->page.nesting[idx].valid_map, s->page.nesting[idx].o_valid_map_offset);
        }
      }
      s->first_output_value = 0;

      // first row within the page to start reading
      if (page_start_row >= min_row) {
        s->first_row = 0;
      } else {
        s->first_row = (int32_t)min(min_row - page_start_row, (size_t)s->page.num_rows);
      }
      // # of rows within the page to read
      s->num_rows = s->page.num_rows;
      if (page_start_row + s->num_rows > min_row + num_rows) {
        s->num_rows = (int32_t)max((int64_t)(min_row + num_rows - page_start_row), INT64_C(0));
      }

      // Find the compressed size of repetition levels
      cur += InitLevelSection(s, cur, end, level_type::REPETITION);
      // Find the compressed size of definition levels
      cur += InitLevelSection(s, cur, end, level_type::DEFINITION);

      s->dict_bits = 0;
      s->dict_base = 0;
      s->dict_size = 0;
      switch (s->page.encoding) {
        case PLAIN_DICTIONARY:
        case RLE_DICTIONARY:
          // RLE-packed dictionary indices, first byte indicates index length in bits
          if (((s->col.data_type & 7) == BYTE_ARRAY) && (s->col.str_dict_index)) {
            // String dictionary: use index
            s->dict_base = reinterpret_cast<const uint8_t *>(s->col.str_dict_index);
            s->dict_size = s->col.page_info[0].num_values * sizeof(nvstrdesc_s);
          } else {
            s->dict_base =
              s->col.page_info[0].page_data;  // dictionary is always stored in the first page
            s->dict_size = s->col.page_info[0].uncompressed_page_size;
          }
          s->dict_run  = 0;
          s->dict_val  = 0;
          s->dict_bits = (cur < end) ? *cur++ : 0;
          if (s->dict_bits > 32 || !s->dict_base) { s->error = (10 << 8) | s->dict_bits; }
          break;
        case PLAIN:
          s->dict_size = static_cast<int32_t>(end - cur);
          s->dict_val  = 0;
          if ((s->col.data_type & 7) == BOOLEAN) { s->dict_run = s->dict_size * 2 + 1; }
          break;
        case RLE: s->dict_run = 0; break;
        default:
          s->error = 1;  // Unsupported encoding
          break;
      }
      if (cur > end) { s->error = 1; }
      s->data_start = cur;
      s->data_end   = end;
    } else {
      s->error = 1;
    }

    s->page_value_count                  = 0;
    s->lvl_count[level_type::REPETITION] = 0;
    s->lvl_count[level_type::DEFINITION] = 0;
    s->nz_count                          = 0;
    s->num_values                        = s->page.num_values;
    s->dict_pos                          = 0;
    s->out_pos                           = 0;

    if (s->col.col_index == column_watch)
     {
       /*
      printf(
        "PAGE SETUP END : page(%d), chunk_idx(%d/%d), col(%d), first_value(%d), "
        "data(%lu)\n",
        page_idx,
        chunk_idx,
        num_chunks,
        s->col.col_index,
        s->first_row,        
        (uint64_t)s->data_out[0]);
        */
    }

    __threadfence_block();
  }
  __syncthreads();

  return true;
}

// now we have to transform that into actual row size information
static __device__ void gpuUpdateValidityOffsetsAndRowIndices(int32_t target_value_count,
                                                             page_state_s *s,
                                                             int t,
                                                             int page_idx,
                                                             bool display)
{          
  // max nesting depth of the column
  int max_depth   = s->col.max_level[level_type::REPETITION];
  // how many (input) values we've processed in the page so far
  int page_value_count = s->page_value_count;  
    
  while(page_value_count < target_value_count){
    // determine the nesting bounds for this thread
    int start_depth = -1;
    int end_depth = -1;
    int d = -1;
    if(page_value_count + t < target_value_count){
      int index = rolling_index(page_value_count + t);
      int r = s->rep[index];    
      start_depth = r;
      d = s->def[index];  
      end_depth = s->page.nesting[d].d_remap;
    }

    // compute the count mask for the first level 
    uint32_t count_mask = BALLOT((0 >= start_depth && 0 <= end_depth) ? 1 : 0);
       
    // always walk from 0 to max_depth even if our start and end depths are different.
    // otherwise we'd have thread/warp synchronization issues on the BALLOT() call.
    for(int s_idx=0; s_idx<=max_depth; s_idx++){
      PageNestingInfo *pni = &s->page.nesting[s_idx];
      int in_range = (s_idx >= start_depth && s_idx <= end_depth) ? 1 : 0;
    
      // everything up to the max_def_level is a real value
      int is_valid = 0;
      if (d >= pni->o_max_def_level && in_range) {
        is_valid = 1;
      }

      // each thread in the warp will set bit T, giving us a mask that tells us total count
      uint32_t valid_count_mask = BALLOT(is_valid);
        
      // if this is the value column emit an index
      if(is_valid && s_idx == max_depth){        
        // Note : mask & ((1 << t) - 1) implies "the count for all threads before me"
        int idx = pni->o_valid_count + __popc(valid_count_mask & ((1 << t) - 1));
        int ofs = pni->o_value_count + __popc(count_mask & ((1 << t) - 1));
        s->nz_idx[rolling_index(idx)] = ofs;
      }

      // compute the count mask for the -next- nesting level. in the case of
      // nested schemas we need this value to generate an offset for this level
      uint32_t next_count_mask = (s_idx < max_depth) ? BALLOT((s_idx+1 >= start_depth && s_idx+1 <= end_depth) ? 1 : 0) : 0;

      // if we're -not- at a leaf column, this is a nested schema, so emit an offset
      if(in_range && s_idx < max_depth){        
        int idx = pni->o_value_count + __popc(count_mask & ((1 << t) - 1));
        (reinterpret_cast<int *>(pni->data_out))[idx] = s->page.nesting[s_idx+1].o_value_count + __popc(next_count_mask & ((1 << t)-1));
      }

      // increment count of valid values and total values
      if(!t){
        pni->o_valid_count += __popc(valid_count_mask);
        pni->o_value_count += __popc(count_mask);
      }

      // update count_mask for the next level down      
      count_mask = next_count_mask;
    }

    page_value_count += min(32, (target_value_count - page_value_count) );
    __syncwarp();    
  }
  
  if(!t){    
    s->nz_count = s->page.nesting[max_depth].o_valid_count;
    s->page_value_count = page_value_count;
  }    
  
  #if 0      
  int max_depth   = s->col.max_level[level_type::REPETITION];
  int value_count = s->page.nesting[max_depth].o_value_count;

  int page_value_count = s->page_value_count;  

  // TODO : make this actually parallel
  if (!t) {
    // index of the field that stores the actual leaf values we're generating indices for
    while (page_value_count < target_value_count) {
      int index = rolling_index(page_value_count);
      if (s->col.col_index == column_watch  && display) {
        //printf("A0\n");
      }
      int r = s->rep[index];
      int d = s->def[index];
      if (s->col.col_index == column_watch && display) {
        //printf("A1\n");
      }

      // repetition level effectively means "nesting depth", which directly corresponds to the
      // output column index. for flat schemas, this value will always be 0.
      int start_depth = r;
      int end_depth   = s->page.nesting[d].d_remap;

      if (s->col.col_index == column_watch && display) {
        //printf("A2\n");
      }

      if (s->col.col_index == column_watch && display) {
        //printf("R/D : %d %d\n", r, d);
      }

      // walk from the start depth to the max depth
      for (int s_idx = start_depth; s_idx <= end_depth; s_idx++) {
        PageNestingInfo *pni = &s->page.nesting[s_idx];
        // printf("pni : %lu\n", pni);

        if (s->col.col_index == column_watch && display) {
          //printf("C0 %d %lu\n", s_idx, s->col.nesting);
        }

        if (d >= pni->o_max_def_level) {
          if (s->col.col_index == column_watch) {
            //            printf("C00 %lu\n", (uint64_t)s->valid_map);
            // printf("C01 %lu\n", (uint64_t)s->valid_map[s_idx]);
          }

          if (pni->valid_map != nullptr) {
            // multiple blocks are working on different pieces of the same column. hence the atomic.  of course, this
            // is the worst possible way to do this.  this is just here as a stopgap until this whole function becomes
            // parallel
            cudf::set_bit(pni->valid_map, pni->o_valid_count + pni->o_valid_map_offset);
          }

          // if we're at the leaf column, we've got a real non-null value
          if (s_idx == max_depth) { 
            s->nz_idx[rolling_index(s->nz_count++)] = value_count;          
          }
        }
        // everything after the end depth is null
        else {
          if (pni->valid_map != nullptr) {
            // multiple blocks are working on different pieces of the same column. hence the atomic.  of course, this
            // is the worst possible way to do this.  this is just here as a stopgap until this whole function becomes
            // parallel
            cudf::clear_bit(pni->valid_map,
                            pni->o_valid_count + pni->o_valid_map_offset);

            // multiple blocks are working on different pieces of the same column. hence the atomic.  of course, this
            // is the worst possible way to do this.  this is just here as a stopgap until this whole function becomes
            // parallel
            pni->o_null_count++;
          }
        }

        if (s_idx == max_depth) { value_count++; }

        // offsets
        if (s_idx != max_depth) {
          (reinterpret_cast<int *>(pni->data_out))[pni->o_value_count] = 
              s->page.nesting[s_idx+1].o_value_count + s->page.nesting[s_idx+1].o_page_start_value;
        }

        pni->o_valid_count++;
        pni->o_value_count++;
      }

      page_value_count++;
    }

    // update page value count
    s->page_value_count = page_value_count;
  
    /*
    if(s->col.col_index == column_watch){
      for(int idx=0; idx<=s->col.max_level[level_type::REPETITION]; idx++){
        printf("null_count[%d] %d\n", idx, s->col.nesting[idx].o_null_count);
        printf("valids[%d]", idx);
        for(int s_idx=0; s_idx<s->col.nesting[idx].o_size; s_idx++){
          printf("%d", bit_is_set(s->valid_map[idx], s_idx) ? 1 : 0);
        }
        printf("\n");
      }

      for(int idx=0; idx<s->col.max_level[level_type::REPETITION]; idx++){
        printf("offsets[%d] ", idx);
        for(int s_idx=0; s_idx<s->col.nesting[idx].o_size; s_idx++){
          printf("%d, ", (reinterpret_cast<int**>(s->data_out))[idx][s_idx]);
        }
        printf("\n");
      }
    }
    */       

    if (s->col.col_index == column_watch  && display) {
      //printf("GUV END : %d, %d %d\n", page_value_count, s->nz_count, s->num_values);
    }
  }
  #endif

  // if we're the terminating page for the column, add the final offset to each level of nesting
  if(!t){
    if (s->page_value_count >= s->num_values && s->page.flags & PAGEINFO_FLAGS_TERMINATOR) {
      for (int s_idx = 0; s_idx <= max_depth; s_idx++) {
        if (s_idx != max_depth) {
          (reinterpret_cast<int *>(s->page.nesting[s_idx].data_out))[s->page.nesting[s_idx].o_value_count] =
            s->page.nesting[s_idx+1].o_value_count + s->page.nesting[s_idx+1].o_page_start_value;
        }
      }
      printf("TERMINATED\n");
    }
  }  

  if(!t && s->col.col_index == column_watch){
    /*
    printf("vals : ");
    for(int idx=0; idx<s->nz_count; idx++){
      printf("%d, ", s->nz_idx[idx]);
    }
    printf("\n");
    */   
    for(int s_idx=0; s_idx<max_depth; s_idx++){
      printf("offsets(0) : ");
      for(int idx=0; idx<=s->page.nesting[s_idx].o_value_count; idx++){
        int offset = (reinterpret_cast<int *>(s->page.nesting[s_idx].data_out))[idx];
        printf("%d, ", offset);
      }
      printf("\n");
    }    
  }
}

// scan through repetition and definition levels, recording validity and
// (int the case of nested types) offset information, until we have produced
// at least target_count non-null values for the data decode step to process
__device__ void gpuDecodeLevels(page_state_s *s, int32_t target_count, int t, int page_idx)
{
  if (!t && s->col.col_index == column_watch) {
    //printf("GDL START, target_count : %d, current count : %d\n", target_count, s->nz_count);
  }
  
  int cur_target_count = target_count;
  while (!s->error && s->nz_count < target_count && s->page_value_count < s->num_values) {
    // decode repetition and definition levels
    gpuDecodeStream(s->rep, s, cur_target_count, t, level_type::REPETITION);
    gpuDecodeStream(s->def, s, cur_target_count, t, level_type::DEFINITION);
    __syncthreads();

    // because the rep and def streams can be encoded differently and we cannot request an exact
    // # of values to be decoded at once (it has to be a multiple of 8) we can only process
    // the lowest # of decoded rep/def levels we have.
    int target_value_count =
      min(s->lvl_count[level_type::REPETITION], s->lvl_count[level_type::DEFINITION]);
    
    // process as much as we can
    gpuUpdateValidityOffsetsAndRowIndices(target_value_count, s, t, page_idx, true);
    cur_target_count += 32;
    __syncthreads();
  }

  if (!t && s->col.col_index == column_watch) {
     //printf("GDL END, target_count : %d, current count : %d\n", target_count, s->nz_count);
  }
}

static __device__ void gpuUpdatePageSizes(page_state_s *s, int32_t target_value_count, int t)
{
  // max nesting depth of the column
  int max_depth   = s->col.max_level[level_type::REPETITION];
  // how many (input) values we've processed in the page so far
  int page_value_count = s->page_value_count;

  // TODO : make this actually parallel    
  /*
  if (!t && s->col.col_index == column_watch) {  
    printf("GUCS START : processing %d new values up to target %d (max depth : %d)\n",
            target_value_count - page_value_count,
            target_value_count,
            max_depth);            
  }  
  */
  while (page_value_count < target_value_count) {
    // determine the nesting bounds for this thread
    int start_depth = -1;
    int end_depth = -1;
    if(page_value_count + t < target_value_count){
      int index = rolling_index(page_value_count + t);
      int r = s->rep[index];    
      start_depth = r;
      int d = s->def[index];  
      end_depth   = s->page.nesting[d].d_remap;
    }

    // increment counts across all nesting depths
    for(int s_idx=0; s_idx<=max_depth; s_idx++){      
      int in_range = (s_idx >= start_depth && s_idx <= end_depth) ? 1 : 0;
      uint32_t count_mask = BALLOT(in_range);      
      if(!t){        
        s->page.nesting[s_idx].o_size += __popc(count_mask);
      }
    }
    page_value_count += min(32, (target_value_count - page_value_count) );
  }

  // update final page value count
  if(!t){
    s->page_value_count = target_value_count;
  }

  // if (s->col.col_index == column_watch) { printf("GUCS END : %d\n", page_value_count); }  

#if 0
  // TODO : make this actually parallel
  if (!t) {
    int max_depth   = s->col.max_level[level_type::REPETITION];
    int value_count = s->page.nesting[max_depth].o_value_count;

    int page_value_count = s->page_value_count;

    if (s->col.col_index == column_watch) {
      /*
      printf("GUCS START : processing %d new values up to target %d (max depth : %d)\n",
             target_value_count - page_value_count,
             target_value_count,
             max_depth);
             */
    }

    while (page_value_count < target_value_count) {
      int index = rolling_index(page_value_count);
      int r     = s->rep[index];
      int d     = s->def[index];

      /*
      if (r == 0) { 
        s->num_rows++;
      }
      */

      // walk from the start depth to the defined depth
      int start_depth = r;
      int end_depth   = s->page.nesting[d].d_remap;
      if (s->col.col_index == column_watch) {
//        printf("R/D : %d %d   S/E %d %d\n", r, d, start_depth, end_depth);
      }
      // multiple blocks are working on different pieces of the same column. hence the atomic.  of course, this
      // is the worst possible way to do this.  this is just here as a stopgap until this whole function becomes
      // parallel
      for (int s_idx = start_depth; s_idx <= end_depth; s_idx++) { 
        s->page.nesting[s_idx].o_size++;
      }
      page_value_count++;
    }

    // update page value count
    s->page_value_count = page_value_count;

    //if (s->col.col_index == column_watch) { printf("GUCS END : %d\n", page_value_count); }
  }
  #endif
}

// blockDim {NTHREADS,1,1}
extern "C" __global__ void __launch_bounds__(NTHREADS)
  gpuComputePageSizes(PageInfo *pages, ColumnChunkDesc *chunks, int32_t num_chunks)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s *const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;

  if (!setupLocalPageInfo(s, pages, chunks, -1, -1, num_chunks)) { return; }

  // setupLocalPageInfo will have filled in the incorrect num_rows computed during
  // header decoding (it will only be valid in the non-nested case), so reset it to
  // zero here
  if (!t) { s->num_rows = 0; }
  __syncthreads();

  if (!t && s->col.col_index == column_watch) { 
    printf("GCC START\n"); 
  }

  // optimization : if gpuDecodeStream were capable of operating with more than 32 threads, we 
  // could go wider here. 
  if (t < 32) {
    // decode in batches of 32. not sure it makes sense to go wider since the decode functions only
    // operate on 1 warp.
    constexpr int batch_size = 32;
    int target_count = batch_size;
    while (!s->error && s->page_value_count < s->num_values) {
      // decode repetition and definition levels. these will attempt to decode at
      // least as up to the target, but may decode a few more. 
      gpuDecodeStream(s->rep, s, target_count, t, level_type::REPETITION);
      gpuDecodeStream(s->def, s, target_count, t, level_type::DEFINITION);
      __syncthreads();

      // we may have decoded different amounts from each stream, so only process what we've been
      int target_value_count =
        min(s->lvl_count[level_type::REPETITION], s->lvl_count[level_type::DEFINITION]);

      // process as much as we can
      gpuUpdatePageSizes(s, target_value_count, t);
      target_count += batch_size;
      __syncthreads();
    }
  }
  // update # rows in the actual page
  if(!t){
    // s->page.num_rows = s->num_rows;    
    s->page.num_rows = s->page.nesting[0].o_size;
  }

  if (!t && s->col.col_index == column_watch) {    
    printf("GCC END\n");
    printf("num rows == %d %d\n", s->page.chunk_idx, s->page.num_rows);    
    int max_depth   = s->col.max_level[level_type::REPETITION];    
    printf("Column %d (max_depth : %d):\n", s->col.col_index, max_depth);    
    for (int idx = 0; idx <= max_depth; idx++) {
      printf("   col_size[%d] : %d\n", idx, s->page.nesting[idx].o_size);
    }        
  }
}

/**
 * @brief Kernel for reading the column data stored in the pages
 *
 * This function will write the page data and the page data's validity to the
 * output specified in the page's column chunk. If necessary, additional
 * conversion will be performed to translate from the Parquet datatype to
 * desired output datatype (ex. 32-bit to 16-bit, string to hash).
 *
 * @param[in] pages List of pages
 * @param[in,out] chunks List of column chunks
 * @param[in] min_row crop all rows below min_row
 * @param[in] num_rows Maximum number of rows to read
 * @param[in] num_chunks Number of column chunks
 **/
// blockDim {NTHREADS,1,1}
extern "C" __global__ void __launch_bounds__(NTHREADS) gpuDecodePageData(PageInfo *pages,
                                                                         ColumnChunkDesc *chunks,
                                                                         size_t min_row,
                                                                         size_t num_rows,
                                                                         int32_t num_chunks,
                                                                         bool has_nesting)
{
  __shared__ __align__(16) page_state_s state_g;

  page_state_s *const s = &state_g;
  int page_idx          = blockIdx.x;
  int t                 = threadIdx.x;
  int out_thread0;

  if (!setupLocalPageInfo(s, pages, chunks, min_row, num_rows, num_chunks)) { return; }

  if (!t) {
    //printf("GDP START : %d, %d, %d, %d, %d\n", page_idx, (int)min_row, (int)num_rows, num_chunks,
     //s->num_values);
  }

  if (s->dict_base) {
    out_thread0 = (s->dict_bits > 0) ? 64 : 32;
  } else {
    out_thread0 =
      ((s->col.data_type & 7) == BOOLEAN || (s->col.data_type & 7) == BYTE_ARRAY) ? 64 : 32;
  }

  while (!s->error && (s->page_value_count < s->num_values || s->out_pos < s->nz_count)) {
    int target_pos;
    int out_pos = s->out_pos;

    if (t < out_thread0) {
      target_pos =
        min(out_pos + 2 * (NTHREADS - out_thread0), s->nz_count + (NTHREADS - out_thread0));
    } else {
      target_pos = min(s->nz_count, out_pos + NTHREADS - out_thread0);
      if (out_thread0 > 32) { target_pos = min(target_pos, s->dict_pos); }
    }
    __syncthreads();
    if (t < 32) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels(s, target_pos, t, page_idx);
    } else if (t < out_thread0) {
      // WARP1: Decode dictionary indices, booleans or string positions
      if (s->dict_base) {
        target_pos = gpuDecodeDictionaryIndices(s, target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BOOLEAN) {
        target_pos = gpuDecodeRleBooleans(s, target_pos, t & 0x1f);
      } else if ((s->col.data_type & 7) == BYTE_ARRAY) {
        gpuInitStringDescriptors(s, target_pos, t & 0x1f);
      }
      if (t == 32) { *(volatile int32_t *)&s->dict_pos = target_pos; }
    } else {      
      // WARP1..WARP3: Decode values
      int dtype = s->col.data_type & 7;
      out_pos += t - out_thread0;
      int output_value_idx = s->nz_idx[rolling_index(out_pos)];

      // nesting level that is storing actual leaf values
      int value_level_index = s->col.max_level[level_type::REPETITION];
      
      if (out_pos < target_pos && output_value_idx >= 0 && s->first_output_value + output_value_idx < s->num_values) {
        uint32_t dtype_len = s->dtype_len;
        uint8_t *dst       = s->page.nesting[value_level_index].data_out + (size_t)output_value_idx * dtype_len;
        if (dtype == BYTE_ARRAY)
          gpuOutputString(s, out_pos, dst, output_value_idx);
        else if (dtype == BOOLEAN)
          gpuOutputBoolean(s, out_pos, dst);
        else if (s->col.converted_type == DECIMAL)
          gpuOutputDecimal(s, out_pos, reinterpret_cast<double *>(dst), dtype);
        else if (dtype == INT96)
          gpuOutputInt96Timestamp(s, out_pos, reinterpret_cast<int64_t *>(dst));
        else if (dtype_len == 8) {
          if (s->ts_scale)
            gpuOutputInt64Timestamp(s, out_pos, reinterpret_cast<int64_t *>(dst));
          else
            gpuOutputFast(s, out_pos, reinterpret_cast<uint2 *>(dst));
        } else if (dtype_len == 4)
          gpuOutputFast(s, out_pos, reinterpret_cast<uint32_t *>(dst));
        else
          gpuOutputGeneric(s, out_pos, dst, dtype_len);
      }
      
      if (t == out_thread0) { *(volatile int32_t *)&s->out_pos = target_pos; }
    }
    __syncthreads();
  }
  __syncthreads();
  if (!t) {
    // Update the number of rows (after cropping to [min_row, min_row+num_rows-1]), and number of
    // valid values

    // ROW VALUE
    // pages[page_idx].num_rows    = s->num_rows - s->first_row;

    // printf("END NULLS : %d %d\n", s->col.nesting[0].null_count, s->col.nesting[1].null_count);
    // pages[page_idx].num_values = s->num_values - s->first_value;

    // pages[page_idx].valid_count = (s->error) ? -s->error : s->page.valid_count;

    if (!t) {
      //printf("GDP END : %d\n", page_idx);
    }
  }
}

// this seems generally useful.  couldn't hurt to see if there's a way to genericize this
// as a general output-to-a-specific-field iterator
struct chunk_row_output_iter {
  PageInfo *p;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type *;
  using reference         = size_type &;
  using iterator_category = thrust::output_device_iterator_tag;

  chunk_row_output_iter operator+ __host__ __device__(int i)
  {
    return chunk_row_output_iter{p + i};
  }
  chunk_row_output_iter operator[] __host__ __device__(int i)
  {
    return chunk_row_output_iter{p + i};
  }
  void operator++ __host__ __device__() { p++; }
  reference operator*__host__ __device__() { return p->chunk_row; }
  void operator= __host__ __device__(value_type v) { p->chunk_row = v; }
};

struct start_offset_output_iterator {
  PageInfo *p;  
  int col_index;
  int nesting_depth;
  int empty = 0;
  using value_type        = size_type;
  using difference_type   = size_type;
  using pointer           = size_type *;
  using reference         = size_type &;
  using iterator_category = thrust::output_device_iterator_tag;

  start_offset_output_iterator operator+ __host__ __device__(int i)
  {    
    return start_offset_output_iterator{p + i, col_index, nesting_depth};
  }
  start_offset_output_iterator operator[] __host__ __device__(int i)
  {
    return start_offset_output_iterator{p + i, col_index, nesting_depth};
  }
  void operator++ __host__ __device__() { p++; }  
  reference operator*__host__ __device__() {             
    if(p->column_idx != col_index || p->flags & PAGEINFO_FLAGS_DICTIONARY){
      return empty;
    }
    return p->nesting[nesting_depth].o_page_start_value;
  }  
  void operator= __host__ __device__(value_type v) {         
    if(p->column_idx == col_index && !(p->flags & PAGEINFO_FLAGS_DICTIONARY)){
      p->nesting[nesting_depth].o_page_start_value = v;
    }
  }
};

cudaError_t PreprocessColumnData(hostdevice_vector<PageInfo>& pages,
                                 hostdevice_vector<ColumnChunkDesc>& chunks,
                                 std::vector<std::vector<std::pair<int, bool>>>& nested_info,
                                 size_t num_rows,
                                 size_t min_row,
                                 cudaStream_t stream)
{
  dim3 dim_block(NTHREADS, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  // computes:
  // per-PageInfo value counts for all levels of nesting.
  // per-PageInfo # of rows
  gpuComputePageSizes<<<dim_grid, dim_block, 0, stream>>>(pages.device_ptr(), chunks.device_ptr(), chunks.size());  

  // computes:
  // output column sizes for each level of nesting
  // per-page start offsets for each level of nesting
  for(size_t idx=0; idx<nested_info.size(); idx++){
    size_t max_depth = nested_info[idx].size()-1;
    for(size_t l_idx=0; l_idx<=max_depth; l_idx++){      
      // column size
      auto page_input = thrust::make_transform_iterator(pages.device_ptr(), [idx, l_idx] __device__ (PageInfo const& page){
        if(page.column_idx != idx || page.flags & PAGEINFO_FLAGS_DICTIONARY){
          return 0;
        }
        return page.nesting[l_idx].o_size;
      });      
      nested_info[idx][l_idx].first = thrust::reduce(rmm::exec_policy(stream)->on(stream), page_input, page_input + pages.size());

      // add 1 for non-leaf levels for the terminating offset
      if(l_idx < max_depth){
        nested_info[idx][l_idx].first++;
      }

      // per-page start offset
      auto key_input = thrust::make_transform_iterator(pages.device_ptr(), [] __device__ (PageInfo const& page){
        return page.column_idx;      
      });
      thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                  key_input,
                                  key_input + pages.size(),
                                  page_input,
                                  start_offset_output_iterator{pages.device_ptr(), static_cast<int>(chunks[idx].col_index), static_cast<int>(l_idx)});
    }    
  }
      
  // computes:
  // PageInfo::chunk_row for all pages
  // !!! This is making the assumption that ordering of pages is sorted by chunk_idx !!!
  auto key_input = thrust::make_transform_iterator(pages.device_ptr(), [] __device__ (PageInfo const& page){
    return page.chunk_idx;
  }); 
  auto page_input = thrust::make_transform_iterator(pages.device_ptr(), [] __device__ (PageInfo const& page){
    return page.num_rows;
  });
  thrust::exclusive_scan_by_key(rmm::exec_policy(stream)->on(stream),
                                key_input,
                                key_input + pages.size(),
                                page_input,
                                chunk_row_output_iter{pages.device_ptr()});    

  PageNestingInfo pni[16];   
  for(size_t idx=0; idx<nested_info.size(); idx++){
    int max_depth = nested_info[idx].size();
    printf("Column %lu info\n", idx);
    for(int l_idx=0; l_idx<max_depth; l_idx++){
      printf("   depth %d : size(%d), nullable(%s)\n", l_idx, nested_info[idx][l_idx].first, nested_info[idx][l_idx].second ? "yes" : "no");
    }
  }  
  for(size_t idx=0; idx<chunks.size(); idx++){
    int max_depth = chunks[idx].max_level[level_type::REPETITION] + 1;
    for(int p_idx=0; p_idx<chunks[idx].num_data_pages; p_idx++){
      PageInfo pi;
      cudaMemcpy(&pi, chunks[idx].page_info + chunks[idx].num_dict_pages + p_idx, sizeof(PageInfo), cudaMemcpyDeviceToHost);      
      cudaMemcpy(pni, pi.nesting, sizeof(PageNestingInfo) * max_depth, cudaMemcpyDeviceToHost);
      printf("Column %d, rg %lu, page %d (%d) (%s), num_values : %d, nested offsets:\n", chunks[idx].col_index, idx, p_idx, pi.column_idx, pi.flags & PAGEINFO_FLAGS_TERMINATOR ? "T" : "", pi.num_values);
      for(int l_idx=0; l_idx<max_depth; l_idx++){
        printf("   depth %d : %d\n", l_idx, pni[l_idx].o_page_start_value);
      }
    }    
  }

  return cudaSuccess;
}

cudaError_t __host__ DecodePageData(PageInfo *pages,
                                    int32_t num_pages,
                                    ColumnChunkDesc *chunks,
                                    int32_t num_chunks,
                                    size_t num_rows,
                                    size_t min_row,
                                    bool has_nesting,
                                    cudaStream_t stream)
{
  dim3 dim_block(NTHREADS, 1);
  dim3 dim_grid(num_pages, 1);  // 1 threadblock per page

  gpuDecodePageData<<<dim_grid, dim_block, 0, stream>>>(
    pages, chunks, min_row, num_rows, num_chunks, has_nesting);

  return cudaSuccess;
}

}  // namespace gpu
}  // namespace parquet
}  // namespace io
}  // namespace cudf