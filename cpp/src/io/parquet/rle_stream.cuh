/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

#pragma once

#include "parquet_gpu.hpp"
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <io/utilities/block_utils.cuh>

#include <thrust/binary_search.h>

namespace cudf::io::parquet::gpu {

// TODO: consider if these should be template parameters to rle_stream
constexpr int num_rle_stream_decode_threads = 512;
// the -1 here is for the look-ahead warp that fills in the list of runs to be decoded
// in an overlapped manner. so if we had 16 total warps:
// - warp 0 would be filling in batches of runs to be processed
// - warps 1-15 would be decoding the previous batch of runs generated
constexpr int num_rle_stream_decode_warps =
  (num_rle_stream_decode_threads / cudf::detail::warp_size) - 1;

constexpr int last_warp_decode_id = num_rle_stream_decode_warps - 1;
constexpr int run_buffer_size = (num_rle_stream_decode_warps * 2);
constexpr int rolling_run_index(int index) { return index % run_buffer_size; }

constexpr int rolling_index(int index) { return index & (2048 - 1); }
/**
 * @brief Read a 32-bit varint integer
 *
 * @param[in,out] cur The current data position, updated after the read
 * @param[in] end The end data position
 *
 * @return The 32-bit value read
 */
inline __device__ uint32_t get_vlq32(uint8_t const*& cur, uint8_t const* end)
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

// a single rle run. may be broken up into multiple rle_batches
template <typename level_t>
struct rle_run {
  int size;               // total size of the run
  int carry;              // values already consumed from previous processing
  int abs_output_pos;     // absolute output position 
  uint8_t const* start;
  int level_run;          // level_run header value
  int first_warp;         // index of the first warp that will process this run
};

// key tuning parameter.  when generating runs, we break them up into
// pieces no larger than run_granularity.  this strikes a balance between:
//
// - time to generate the runs
// vs
// - time to process the runs
//
// this warp generates runs while the remaining warps decode them. A granularity of 32
// would mean each warp processes 32 values, or 1 decode "pass". if this process finishes
// before the next fill_run_batch() completes, the gpu ends up bottlenecked on this warp. so we 
// want to feed enough work to the decode warps (larger run_granularity) such that we have 
// time to complete generating the next set of runs.  Conversely, we don't want to send too
// much work to the each decode warp because then we potentially end up underutilizing them 
// (for example, if we had an output limit of 2048 and we set the granularity to 512, we would only
// generate 4 decode warps of work.
//
//
constexpr int run_granularity = 96;

// a stream of rle_runs
template <typename level_t>
struct rle_stream {
  int level_bits;
  int level_bytes;
  uint8_t const* start;  
  uint8_t const* end;

  int max_output_values;
  int total_values;

  level_t* output;
  rle_run<level_t>* runs;

  // fill warp
  uint8_t const* f_cur;
  int f_window_start, f_window_end;
  int f_next_window_start, f_next_window_end;
  int f_run_index;
  int f_warp_index;
  int f_output_pos;  
    
  //int output_pos;
  //int run_index;
  //int fill_warp_index;
  //int warp_index;
  //int window_start;
  //int window_end;
  //int next_window_start;
  //int next_window_end;

  // decode warps
  int dc_warp_index;
  int dc_cur_values;


  __device__ rle_stream(rle_run<level_t>* _runs) : runs(_runs) {}
  __device__ void init(int _level_bits,
                       uint8_t const* _start,
                       uint8_t const* _end,
                       int _max_output_values,
                       level_t* _output,
                       int _total_values)
  {
    level_bits = _level_bits;
    level_bytes = (level_bits + 7) >> 3;
    start      = _start;
    end        = _end;

    max_output_values = _max_output_values;
    output            = _output;
  

    total_values = _total_values;

    f_cur = start;
    f_window_start = 0;
    f_window_end = 0;
    f_next_window_start = 0;
    f_next_window_end = 0;
    f_run_index = 0;
    f_warp_index = 0;
    f_output_pos = 0;

    dc_warp_index = 0;
    dc_cur_values = 0;
  }  
  
  __device__ inline int decode_next(int t, int count = -1, int roll = 0, bool print = false)
  {
    int const output_count = count < 0 ? min(max_output_values, (total_values - dc_cur_values)) : count;

    // special case. if level_bits == 0, just return all zeros. this should tremendously speed up
    // a very common case: columns with no nulls, especially if they are non-nested.
    // TODO: should we also attempt to use this code for very long runs of repeated values?
    if (level_bits == 0) {
      return decode_zeros(t, output_count);
    }

    // otherwise, full decode.
    int const warp_id        = t / cudf::detail::warp_size;
    int const warp_decode_id = warp_id - 1;
    int const warp_lane      = t % cudf::detail::warp_size;

    // iterator for computing batch decode boundaries.
    auto r_index = thrust::make_transform_iterator(thrust::make_counting_iterator(0), [&] __device__ (int i) -> int {
      return runs[rolling_run_index(i)].first_warp;
    });

    __shared__ int run_window_start;
    __shared__ int run_window_end;
    // __shared__ int next_window_start;
    __shared__ int next_warp_index;
    __shared__ int values_processed;
    if (!t) {
      // carryover from the last call.
      // thrust::tie(run_window_start, run_window_end) = get_next_window();
      // next_window_start = run_window_start;
      run_window_start = f_window_start;
      run_window_end = get_next_window(run_window_start);
      values_processed = 0;
      next_warp_index = dc_warp_index;
    }
    __syncthreads();
        
    int run_window_size = run_window_end - run_window_start;
    do {
      // warp 0 reads ahead and generates batches of runs to be decoded by remaining warps.
      if (!warp_id) {
        // keep the run start buffer full
        if(!warp_lane){
          fill_runs(print);
        }
      }
      // remaining warps decode the runs
      else if(run_window_size > 0){
        int const warp_index = dc_warp_index + warp_decode_id;

        // figure out what run we are processing
        int _run_index = thrust::lower_bound(thrust::seq,
                                            r_index + run_window_start,
                                            r_index + run_window_end,
                                            warp_index) - r_index;
        int run_index = (_run_index < run_window_end) && (r_index[_run_index] == warp_index) ? _run_index : _run_index - 1;
        /*
        if(!warp_lane){
          if(run_index < run_window_start){
            printf("BAD INDEX (%d %d) (%d %d) (%d %d)\n", run_index, _run_index, run_window_start, run_window_end, warp_index, r_index[run_index]);
          }
        }
        */
        auto& run  = runs[rolling_run_index(run_index)];
        
        // decode our piece of the run
        int const run_offset = ((warp_index - run.first_warp) * run_granularity) + (warp_decode_id == 0 ? run.carry : 0);
        int const output_offset = (run.abs_output_pos + run_offset) - dc_cur_values;
        
        /*
        if(!warp_lane){
          printf("D0(%d, %d): %d(%d %d) %d %d (%d %d)\n", warp_index, run.first_warp, run_index, run.size, run_offset, output_offset, run.carry, run_window_start, run_window_end);          
        }
        */        

        if(output_offset < output_count){
          int const run_remaining = run.size - run_offset;
          int const output_remaining = output_count - output_offset;
          int const ideal_size = min(run_remaining, run_granularity);
          int const size = min(ideal_size, output_remaining);
          
          /*
          if(!warp_lane){
            printf("D1(%d, %d): %d %d %d %d\n", warp_index, run_index, run_remaining, output_remaining, ideal_size, size);
          } 
          */
          
          decode(warp_lane, run.level_run, run.start, output + output_offset, run_offset, size, roll);

          // last warp to write anything updates the results
          int const end = output_offset + size;
          if(!warp_lane && (end == output_count || warp_decode_id == last_warp_decode_id)){
            values_processed = end;
            if(size < ideal_size){
              run.carry = size;
              next_warp_index = warp_index;
            } else {
              next_warp_index = warp_index + 1;
            }
            // next_window_start = size < run_remaining ? run_index : run_index + 1;
            run_window_start = size < run_remaining ? run_index : run_index + 1;

            /*                
            printf("D2(%d, %d): (%d %d) (%d %d %d %d)\n", warp_index, run_index, 
                                                size, run_remaining,
                                                values_processed, next_warp_index, next_window_start, run.carry);
            */
          }
        }
      }

      __syncthreads();
      // if we haven't run out of space, retrieve the next batch. otherwise leave it for the next
      // call.
      if (!t){
        /*
        if(values_processed < output_count) {
          thrust::tie(run_window_start, run_window_end) = get_next_window();
        }
        // update window start so the next call to fill_runs() has work to do
        f_window_start = next_window_start;
        */
        run_window_end = get_next_window(run_window_start);

        // printf("W: %d <-> %d\n", run_window_start, run_window_end);
      }
      dc_warp_index = next_warp_index;
      
      __syncthreads();
      run_window_size = run_window_end - run_window_start;
    } while (run_window_size > 0 && values_processed < output_count);

    dc_cur_values += values_processed; 
        
    /*
    if(!t && count >= 0){
      printf("%d / %d --------------------\n", dc_cur_values, total_values);
    }
    __syncthreads();
    */

    // valid for every thread
    return values_processed;
  }

private:
  __device__ inline int decode_zeros(int t, int output_count)
  {
    int written = 0;
    while (written < output_count) {
      int const batch_size = min(num_rle_stream_decode_threads, output_count - written);
      if (t < batch_size) {
        output[written + t] = 0; 
      }
      written += batch_size;
    }
    dc_cur_values += output_count;
    return output_count;
  }

  __device__ inline int get_next_window(int decode_start)
  {
    f_window_start = decode_start;
    return f_next_window_end;
  }

  // fill in up to num_rle_stream_decode_warps runs or until we reach the max_count limit.
  // this function is the critical hotspot.  please be very careful altering it.  
  __device__ void fill_runs(bool print)
  {    
    // f_window_start is updated by the decode warps.
    f_next_window_start = f_window_start;
    int window_size = f_window_end - f_window_start;
    int to_fill = min(num_rle_stream_decode_warps, run_buffer_size - window_size);
    while(to_fill && f_cur < end){
      // Encoding::RLE
      // bytes for the varint header
      int const input_level_run = get_vlq32(f_cur, end);
      
      rle_run<level_t>& r = runs[rolling_run_index(f_run_index)];
      r.start = f_cur;

      int input_run_size;
      if (input_level_run & 1) {
        input_run_size  = (input_level_run >> 1) * 8;
        int const run_size8 = (input_run_size + 7) >> 3;
        f_cur += run_size8 * level_bits;
      }
      // repeated value run
      else {
        input_run_size = (input_level_run >> 1);
        f_cur += level_bytes;
      }        
      r.size = input_run_size;
      r.abs_output_pos = f_output_pos;
      r.level_run = input_level_run;
      r.first_warp = f_warp_index;
      r.carry = 0;
      f_warp_index += (input_run_size + (run_granularity - 1)) / run_granularity;
      f_output_pos += input_run_size;
      f_window_end++; 
      f_run_index++;
      to_fill--;
      // printf("R(%d): %d %d %d\n", f_run_index - 1, r.first_warp, r.abs_output_pos, r.size);
    }

    f_next_window_end = f_window_end;
  }

  __device__ void decode(int lane,
                        int level_run,
                        uint8_t const* run_start,
                        level_t* output,
                        int run_offset,
                        int size,
                        int roll)
  {
    int output_pos = 0;
    int remain     = size;

    // for bitpacked/literal runs, total size is always a multiple of 8. so we need to take care if
    // we are not starting/ending exactly on a run boundary
    uint8_t const* run_cur;
    int level_val;
    if (level_run & 1) {
      int const effective_offset = cudf::util::round_down_safe(run_offset, 8);
      int const lead_values      = (run_offset - effective_offset);
      output_pos -= lead_values;
      remain += lead_values;
      run_cur = run_start + ((effective_offset >> 3) * level_bits);
    }
    // if this is a repeated run, compute the repeated value 
    else {
      level_val = run_start[0];
      if (level_bytes > 1){
        level_val |= run_start[1] << 8; 
        if(level_bytes > 2){
          level_val |= run_start[2] << 16;
        }
      }
      level_val &= ((1 << level_bits) - 1);
    }

    // process
    while (remain > 0) {
      int const batch_len = min(32, remain);

      // if this is a literal run. each thread computes its own level_val
      if (level_run & 1) {
        int const batch_len8 = (batch_len + 7) >> 3;
        if (lane < batch_len) {
          int bitpos                = lane * level_bits;
          uint8_t const* cur_thread = run_cur + (bitpos >> 3);
          bitpos &= 7;
          level_val = 0;
          if (cur_thread < end) {
            uint32_t c = 8 - bitpos;
            level_val   = (*cur_thread++) >> bitpos;
            if (c < level_bits && cur_thread < end) {
              level_val |= (*cur_thread++) << c;
              c += 8;
              if (c < level_bits && cur_thread < end) {
                level_val |= (*cur_thread++) << c;
                c += 8;
                if (c < level_bits && cur_thread < end) { 
                  level_val |= (*cur_thread) << c; 
                }
              }
            }
            level_val &= (1 << level_bits) - 1;
          }
        }

        run_cur += batch_len8 * level_bits;
      }

      // store level_val
      if (lane < batch_len && (lane + output_pos) >= 0) {
        output[rolling_index(lane + output_pos + roll)] = level_val;
      }
      remain -= batch_len;
      output_pos += batch_len;
    }
  }
};

}  // namespace cudf::io::parquet::gpu