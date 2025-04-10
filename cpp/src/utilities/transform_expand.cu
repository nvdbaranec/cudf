/**
 * @brief Utility function which expands a range of sizes, and invokes a function on
 * each element in all of the groups, providing group and subgroup indices, returning
 * the generated list of values.
 *
 * As an example, imagine we had the input
 * [2, 0, 5, 1]
 *
 * transform_expand will invoke `op` 8 times, with the values
 *
 * (0, 0) (0, 1),  (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),   (3, 0)
 *
 * @param first Beginning of the input range
 * @param end End of the input range
 * @param op Function to be invoked on each expanded value. This function should return
 * a single value that gets collected into the overall result returned from transform_expand
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr User provided resource used for allocating the returned device memory
 *
 * @return The resulting device_uvector containing the output from `op`
 */
template <typename SizeIterator, typename GroupFunction>
rmm::device_uvector<std::invoke_result_t<GroupFunction>> transform_expand(
  SizeIterator first,
  SizeIterator last,
  GroupFunction op,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  auto temp_mr = cudf::get_current_device_resource_ref();

  auto value_count  = std::distance(first, last);
  auto size_wrapper = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_t>([value_count, first] __device__(size_t i) {
      return i >= value_count ? 0 : first[i];
    }));
  rmm::device_uvector<size_t> group_offsets(value_count + 1, stream, temp_mr);
  thrust::exclusive_scan(rmm::exec_policy(stream, temp_mr),
                         size_wrapper,
                         size_wrapper + group_offsets.size(),
                         group_offsets.begin());
  size_t total_size = group_offsets.back_element(stream);  // note memcpy and device sync

  using OutputType = std::invoke_result_t<GroupFunction>;
  rmm::device_uvector<OutputType> result(total_size, stream, mr);
  auto iter = thrust::make_counting_iterator(0);
  thrust::transform(rmm::exec_policy(stream, temp_mr),
                    iter,
                    iter + total_size,
                    result.begin(),
                    cuda::proclaim_return_type<OutputType>(
                      [op,
                       group_offsets_begin = group_offsets.begin(),
                       group_offsets_end   = group_offsets.end()] __device__(size_t i) {
                        auto const index =
                          thrust::upper_bound(
                            thrust::seq, group_offsets_begin, group_offsets_end, i) -
                          group_offsets_begin;
                        auto const group_index = i < group_offsets_begin[index] ? index - 1 : index;
                        auto const intra_group_index = i - group_offsets_begin[group_index];
                        return op(group_index, intra_group_index);
                      }));

  return result;
}
