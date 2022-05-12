/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/combine.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/iterator/discard_iterator.h>

#include <cudf_test/column_utilities.hpp>

namespace cudf {
namespace lists {
namespace detail {

namespace {

/**
 * @brief Generates the new set of offsets that regroups the concatenated-by-column inputs
 * into concatenated-by-rows inputs, and the associated null mask.
 *
 * If we have the following input columns
 *
 * s1 = [{0, 1}, {2, 3, 4}, {5}, {},           {6, 7}]
 * s2 = [{8},    {9},       {},  {10, 11, 12}, {13, 14, 15, 16}]
 *
 * We can rearrange the child data using a normal concatenate and a gather such that
 * the resulting values are in the correct order.  For the above example, the
 * child column would look like:
 *
 * {0, 1, 8, 2, 3, 4, 9, 5, 10, 11, 12, 6, 7, 13, 14, 15}
 *
 * Because we did a regular concatenate (and a subsequen gather to reorder the rows),
 * the rows of the lists would look like
 *
 * [{0, 1}, {8}, {2, 3, 4}, {9}, {5}, {10, 11, 12}, {6, 7}, {13, 14, 15, 16}]
 *
 * What we really want is
 *
 * [{0, 1, 8}, {2, 3, 4, 9}, {5}, {10, 11, 12}, {6, 7, 13, 14, 15, 16}]
 *
 * We can do this just be recomputing a new offsets column that regroups things.
 *
 */
std::tuple<std::unique_ptr<column>, rmm::device_buffer, size_type>
generate_regrouped_offsets_and_null_mask(table_view const& input,
                                         bool input_contains_nulls,
                                         concatenate_null_policy null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  auto tdv = table_device_view::create(input, stream);

  // outgoing offsets.
  auto offsets = cudf::make_fixed_width_column(data_type{type_to_id<offset_type>()},
                                               input.num_rows() + 1,
                                               mask_state::UNALLOCATED,
                                               stream,
                                               mr);

  // generate sizes for the regrouped rows
  auto keys = cudf::detail::make_counting_transform_iterator(
    0, [num_columns = input.num_columns()] __device__(size_type i) { return i / num_columns; });
  auto values = cudf::detail::make_counting_transform_iterator(
    0, [num_columns = input.num_columns(), tdv = *tdv] __device__(size_type i) {
      auto const col_index = i % num_columns;
      auto const row_index = i / num_columns;
      auto offsets =
        tdv.column(col_index).child(lists_column_view::offsets_column_index).data<offset_type>() +
        tdv.column(col_index).offset();
      return offsets[row_index + 1] - offsets[row_index];
    });
  thrust::reduce_by_key(rmm::exec_policy(stream),
                        keys,
                        keys + (input.num_rows() * input.num_columns()),
                        values,
                        thrust::make_discard_iterator(),
                        offsets->mutable_view().begin<offset_type>());

  // convert to offsets
  thrust::exclusive_scan(rmm::exec_policy(stream),
                         offsets->view().begin<offset_type>(),
                         offsets->view().begin<offset_type>() + input.num_rows() + 1,
                         offsets->mutable_view().begin<offset_type>(),
                         0);

  // generate appropriate null mask
  auto [null_mask, null_count] = [&]() {
    // if the input doesn't contain nulls, no work to do
    if (!input_contains_nulls) {
      return std::pair<rmm::device_buffer, size_type>{rmm::device_buffer{}, 0};
    }

    // generate the total # of nulls for each input row
    rmm::device_uvector<size_type> null_counts(input.num_rows(), stream);
    auto null_values =
      cudf::detail::make_counting_transform_iterator(0, [tdv = *tdv] __device__(size_type i) {
        auto const col_index = i % tdv.num_columns();
        auto const row_index = i / tdv.num_columns();
        auto const& col      = tdv.column(col_index);
        return col.null_mask() ? (bit_is_set(col.null_mask(), row_index + col.offset()) ? 0 : 1)
                               : 0;
      });
    thrust::reduce_by_key(rmm::exec_policy(stream),
                          keys,
                          keys + (input.num_rows() * input.num_columns()),
                          null_values,
                          thrust::make_discard_iterator(),
                          null_counts.data());

    // row is null if -all- input rows are null
    if (null_policy == concatenate_null_policy::IGNORE) {
      auto is_valid = [num_columns = input.num_columns()] __device__(size_type null_count) {
        return null_count == num_columns ? 0 : 1;
      };
      return cudf::detail::valid_if(
        null_counts.begin(), null_counts.begin() + input.num_rows(), is_valid, stream, mr);
    }

    // row is null if -any- input rows are null
    auto is_valid = [] __device__(size_type null_count) { return null_count > 0 ? 0 : 1; };
    return cudf::detail::valid_if(
      null_counts.begin(), null_counts.begin() + input.num_rows(), is_valid, stream, mr);
  }();

  return {std::move(offsets), std::move(null_mask), null_count};
}

}  // anonymous namespace

/**
 * @copydoc cudf::lists::concatenate_rows
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         rmm::cuda_stream_view stream,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_EXPECTS(input.num_columns() > 0, "The input table must have at least one column.");

  auto const entry_type = lists_column_view(*input.begin()).child().type();
  CUDF_EXPECTS(
    std::all_of(input.begin(),
                input.end(),
                [](column_view const& col) { return col.type().id() == cudf::type_id::LIST; }),
    "All columns of the input table must be of lists column type.");
  CUDF_EXPECTS(
    std::all_of(std::next(input.begin()),
                input.end(),
                [a = *input.begin()](column_view const& b) { return column_types_equal(a, b); }),
    "The types of entries in the input columns must be the same.");

  auto const num_rows = input.num_rows();
  auto const num_cols = input.num_columns();
  if (num_rows == 0) { return cudf::empty_like(input.column(0)); }
  if (num_cols == 1) { return std::make_unique<column>(*(input.begin()), stream, mr); }

  // concatenate the input table into one column.
  // TODO: if we had a multi-column gather implementation, this step would be
  // unnecessary.
  std::vector<column_view> cols(input.num_columns());
  std::copy(input.begin(), input.end(), cols.begin());
  auto concat = cudf::detail::concatenate(cols, stream);

  // perform the gather to rearrange the rows in desired child order. this will produce -almost-
  // what we want. the data of the children will be exactly what we want, but will be grouped as if
  // we had concatenated all the rows together instead of concatenating within the rows.  to fix
  // this we can simply swap in a new set of offsets that re-groups them.
  auto iter = cudf::detail::make_counting_transform_iterator(
    0, [num_columns = input.num_columns(), num_rows = input.num_rows()] __device__(size_type i) {
      auto const src_col_index    = i % num_columns;
      auto const src_row_index    = i / num_columns;
      auto const concat_row_index = (src_col_index * num_rows) + src_row_index;
      return concat_row_index;
    });
  auto gathered = cudf::detail::gather(table_view({*concat}),
                                       iter,
                                       iter + (input.num_columns() * input.num_rows()),
                                       out_of_bounds_policy::DONT_CHECK,
                                       stream,
                                       mr);

  // generate regrouped offsets and null mask
  auto [offsets, null_mask, null_count] =
    generate_regrouped_offsets_and_null_mask(input, concat->has_nulls(), null_policy, stream, mr);

  // reassemble the underlying child data with the regrouped offsets and null mask
  column& col   = gathered->get_column(0);
  auto contents = col.release();
  return cudf::make_lists_column(
    input.num_rows(),
    std::move(offsets),
    std::move(contents.children[lists_column_view::child_column_index]),
    null_count,
    std::move(null_mask),
    stream,
    mr);
}

}  // namespace detail

/**
 * @copydoc cudf::lists::concatenate_rows
 */
std::unique_ptr<column> concatenate_rows(table_view const& input,
                                         concatenate_null_policy null_policy,
                                         rmm::mr::device_memory_resource* mr)
{
  CUDF_FUNC_RANGE();
  return detail::concatenate_rows(input, null_policy, rmm::cuda_stream_default, mr);
}

}  // namespace lists
}  // namespace cudf
