/*
 * Copyright (c) 2019, NVIDIA CORPORATION.
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

/**
 * @file reader_impl.cu
 * @brief cuDF-IO Parquet reader class implementation
 */

#include "reader_impl.hpp"

#include <io/comp/gpuinflate.h>

#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/thrust_rmm_allocator.h>
#include <rmm/device_buffer.hpp>

#include <algorithm>
#include <array>
#include <numeric>
#include <regex>

#include "printer.hpp"

namespace cudf {
namespace io {
namespace detail {
namespace parquet {
// Import functionality that's independent of legacy code
using namespace cudf::io::parquet;
using namespace cudf::io;

namespace {

type_id to_type_id(SchemaElement const &schema,
                   bool strings_to_categorical,
                   type_id timestamp_type_id)
{
  parquet::Type physical         = schema.type;
  parquet::ConvertedType logical = schema.converted_type;
  int32_t decimal_scale          = schema.decimal_scale;

  // printf("PQT : %d / %d\n", (int)physical, (int)logical);

  // Logical type used for actual data interpretation; the legacy converted type
  // is superceded by 'logical' type whenever available.
  switch (logical) {
    case parquet::UINT_8: return type_id::UINT8;
    case parquet::INT_8: return type_id::INT8;
    case parquet::UINT_16: return type_id::UINT16;
    case parquet::INT_16: return type_id::INT16;
    case parquet::UINT_32: return type_id::UINT32;
    case parquet::UINT_64: return type_id::UINT64;
    case parquet::DATE: return type_id::TIMESTAMP_DAYS;
    case parquet::TIMESTAMP_MICROS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MICROSECONDS;
    case parquet::TIMESTAMP_MILLIS:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_MILLISECONDS;
    case parquet::DECIMAL:
      if (decimal_scale != 0 || (physical != parquet::INT32 && physical != parquet::INT64)) {
        return type_id::FLOAT64;
      }
      break;

    case parquet::LIST: return type_id::LIST;

    default: break;
  }

  // Physical storage type supported by Parquet; controls the on-disk storage
  // format in combination with the encoding type.
  switch (physical) {
    case parquet::BOOLEAN: return type_id::BOOL8;
    case parquet::INT32: return type_id::INT32;
    case parquet::INT64: return type_id::INT64;
    case parquet::FLOAT: return type_id::FLOAT32;
    case parquet::DOUBLE: return type_id::FLOAT64;
    case parquet::BYTE_ARRAY:
    case parquet::FIXED_LEN_BYTE_ARRAY:
      // Can be mapped to INT32 (32-bit hash) or STRING
      return strings_to_categorical ? type_id::INT32 : type_id::STRING;
    case parquet::INT96:
      return (timestamp_type_id != type_id::EMPTY) ? timestamp_type_id
                                                   : type_id::TIMESTAMP_NANOSECONDS;
    default: break;
  }

  return type_id::EMPTY;
}

/**
 * @brief Function that translates cuDF time unit to Parquet clock frequency
 */
constexpr int32_t to_clockrate(type_id timestamp_type_id)
{
  switch (timestamp_type_id) {
    case type_id::TIMESTAMP_SECONDS: return 1;
    case type_id::TIMESTAMP_MILLISECONDS: return 1000;
    case type_id::TIMESTAMP_MICROSECONDS: return 1000000;
    case type_id::TIMESTAMP_NANOSECONDS: return 1000000000;
    default: return 0;
  }
}

/**
 * @brief Function that returns the required the number of bits to store a value
 */
template <typename T = uint8_t>
T required_bits(uint32_t max_level)
{
  return static_cast<T>(CompactProtocolReader::NumRequiredBits(max_level));
}

std::tuple<int32_t, int32_t, int8_t> conversion_info(type_id column_type_id,
                                                     type_id timestamp_type_id,
                                                     parquet::Type physical,
                                                     int8_t converted,
                                                     int32_t length)
{
  int32_t type_width = (physical == parquet::FIXED_LEN_BYTE_ARRAY) ? length : 0;
  int32_t clock_rate = 0;
  if (column_type_id == type_id::INT8 or column_type_id == type_id::UINT8) {
    type_width = 1;  // I32 -> I8
  } else if (column_type_id == type_id::INT16 or column_type_id == type_id::UINT16) {
    type_width = 2;  // I32 -> I16
  } else if (column_type_id == type_id::INT32) {
    type_width = 4;  // str -> hash32
  } else if (is_timestamp(data_type{column_type_id})) {
    clock_rate = to_clockrate(timestamp_type_id);
  }

  printf("converted type : %d\n", converted);
  int8_t converted_type = converted;
  if (converted_type == parquet::DECIMAL && column_type_id != type_id::FLOAT64) {
    converted_type = parquet::UNKNOWN;  // Not converting to float64
  }
  return std::make_tuple(type_width, clock_rate, converted_type);
}

}  // namespace

std::string name_from_path(const std::vector<std::string> &path_in_schema)
{
  std::string s = (path_in_schema.size() > 0) ? path_in_schema[0] : "";
  for (size_t i = 1; i < path_in_schema.size(); i++) { s += "." + path_in_schema[i]; }
  return s;
}

/**
 * @brief Class for parsing dataset metadata
 */
struct metadata : public FileMetaData {
  explicit metadata(datasource *source)
  {
    constexpr auto header_len = sizeof(file_header_s);
    constexpr auto ender_len  = sizeof(file_ender_s);

    const auto len           = source->size();
    const auto header_buffer = source->host_read(0, header_len);
    const auto header        = (const file_header_s *)header_buffer->data();
    const auto ender_buffer  = source->host_read(len - ender_len, ender_len);
    const auto ender         = (const file_ender_s *)ender_buffer->data();
    CUDF_EXPECTS(len > header_len + ender_len, "Incorrect data source");
    CUDF_EXPECTS(header->magic == PARQUET_MAGIC && ender->magic == PARQUET_MAGIC,
                 "Corrupted header or footer");
    CUDF_EXPECTS(ender->footer_len != 0 && ender->footer_len <= (len - header_len - ender_len),
                 "Incorrect footer length");

    const auto buffer = source->host_read(len - ender->footer_len - ender_len, ender->footer_len);
    CompactProtocolReader cp(buffer->data(), ender->footer_len);
    CUDF_EXPECTS(cp.read(this), "Cannot parse metadata");
    CUDF_EXPECTS(cp.InitSchema(this), "Cannot initialize schema");
  }
};

class aggregate_metadata {
  std::vector<metadata> const per_file_metadata;
  std::map<std::string, std::string> const agg_keyval_map;
  size_type const num_rows;
  size_type const num_row_groups;
  std::vector<std::string> const column_names;
  /**
   * @brief Create a metadata object from each element in the source vector
   */
  auto metadatas_from_sources(std::vector<std::unique_ptr<datasource>> const &sources)
  {
    std::vector<metadata> metadatas;
    std::transform(
      sources.cbegin(), sources.cend(), std::back_inserter(metadatas), [](auto const &source) {
        return metadata(source.get());
      });
    return metadatas;
  }

  /**
   * @brief Merge the keyvalue maps from each per-file metadata object into a single map.
   */
  auto merge_keyval_metadata()
  {
    std::map<std::string, std::string> merged;
    // merge key/value maps TODO: warn/throw if there are mismatches?
    for (auto const &pfm : per_file_metadata) {
      for (auto const &kv : pfm.key_value_metadata) { merged[kv.key] = kv.value; }
    }
    return merged;
  }

  /**
   * @brief Sums up the number of rows of each source
   */
  size_type calc_num_rows() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto &sum, auto &pfm) {
        return sum + pfm.num_rows;
      });
  }

  /**
   * @brief Sums up the number of row groups of each source
   */
  size_type calc_num_row_groups() const
  {
    return std::accumulate(
      per_file_metadata.begin(), per_file_metadata.end(), 0, [](auto &sum, auto &pfm) {
        return sum + pfm.row_groups.size();
      });
  }
  std::vector<std::string> gather_column_names()
  {
    for (auto const &pfm : per_file_metadata) {
      if (pfm.row_groups.size() != 0) {
        std::vector<std::string> column_names;
        for (const auto &chunk : pfm.row_groups[0].columns) {
          column_names.emplace_back(name_from_path(chunk.meta_data.path_in_schema));
        }
        return column_names;
      }
    }
    return {};
  }

 public:
  aggregate_metadata(std::vector<std::unique_ptr<datasource>> const &sources)
    : per_file_metadata(metadatas_from_sources(sources)),
      agg_keyval_map(merge_keyval_metadata()),
      num_rows(calc_num_rows()),
      num_row_groups(calc_num_row_groups()),
      column_names(gather_column_names())
  {
    // Verify that the input files have matching numbers of columns
    size_type num_cols = -1;
    for (auto const &pfm : per_file_metadata) {
      if (pfm.row_groups.size() != 0) {
        if (num_cols == -1)
          num_cols = pfm.row_groups[0].columns.size();
        else
          CUDF_EXPECTS(num_cols == static_cast<size_type>(pfm.row_groups[0].columns.size()),
                       "All sources must have the same number of columns");
      }
    }
    // Verify that the input files have matching schemas
    for (auto const &pfm : per_file_metadata) {
      CUDF_EXPECTS(per_file_metadata[0].schema == pfm.schema,
                   "All sources must have the same schemas");
    }
  }

  auto const &get_row_group(size_type idx, size_type src_idx) const
  {
    CUDF_EXPECTS(src_idx >= 0 && src_idx < static_cast<size_type>(per_file_metadata.size()),
                 "invalid source index");
    return per_file_metadata[src_idx].row_groups[idx];
  }

  auto get_num_rows() const { return num_rows; }

  auto get_num_row_groups() const { return num_row_groups; }

  auto const &get_schema(int idx) const { return per_file_metadata[0].schema[idx]; }

  auto const &get_key_value_metadata() const { return agg_keyval_map; }

  inline SchemaElement const &get_column_schema(int col_index) const
  {
    auto &pfm = per_file_metadata[0];
    return pfm.schema[pfm.row_groups[0].columns[col_index].schema_idx];
  }

  inline int get_column_leaf_schema_index(int col_index) const
  {
    return per_file_metadata[0].row_groups[0].columns[col_index].leaf_schema_idx;
  }

  inline SchemaElement const &get_column_leaf_schema(int col_index) const
  {
    return per_file_metadata[0].schema[get_column_leaf_schema_index(col_index)];
  }

  inline int get_nesting_depth(int col_index)
  {
    auto &pfm = per_file_metadata[0];

    // this is not strictly correct.  parquet supports repeated fields for non-lists.
    // we are making the assumption that repeated fields here indicate lists, specifically.
    // this will need to be adjusted for struct support
    // see : the "Nested Types" section here
    // https://github.com/apache/parquet-format/blob/master/LogicalTypes.md

    int index = get_column_leaf_schema_index(col_index);
    int depth = 0;

    // walk upwards, skipping repeated fields
    while (index > 0) {
      if (pfm.schema[index].repetition_type != REPEATED) { depth++; }
      index = pfm.schema[index].parent_idx;
    }

    return depth;
  }

  /**
   * @brief Extracts the pandas "index_columns" section
   *
   * PANDAS adds its own metadata to the key_value section when writing out the
   * dataframe to a file to aid in exact reconstruction. The JSON-formatted
   * metadata contains the index column(s) and PANDA-specific datatypes.
   *
   * @return comma-separated index column names in quotes
   */
  std::string get_pandas_index() const
  {
    auto it = agg_keyval_map.find("pandas");
    if (it != agg_keyval_map.end()) {
      // Captures a list of quoted strings found inside square brackets after `"index_columns":`
      // Inside quotes supports newlines, brackets, escaped quotes, etc.
      // One-liner regex:
      // "index_columns"\s*:\s*\[\s*((?:"(?:|(?:.*?(?![^\\]")).?)[^\\]?",?\s*)*)\]
      // Documented below.
      std::regex index_columns_expr{
        R"("index_columns"\s*:\s*\[\s*)"  // match preamble, opening square bracket, whitespace
        R"(()"                            // Open first capturing group
        R"((?:")"                         // Open non-capturing group match opening quote
        R"((?:|(?:.*?(?![^\\]")).?))"     // match empty string or anything between quotes
        R"([^\\]?")"                      // Match closing non-escaped quote
        R"(,?\s*)"                        // Match optional comma and whitespace
        R"()*)"                           // Close non-capturing group and repeat 0 or more times
        R"())"                            // Close first capturing group
        R"(\])"                           // Match closing square brackets
      };
      std::smatch sm;
      if (std::regex_search(it->second, sm, index_columns_expr)) { return std::move(sm[1].str()); }
    }
    return "";
  }

  /**
   * @brief Extracts the column name(s) used for the row indexes in a dataframe
   *
   * @param names List of column names to load, where index column name(s) will be added
   */
  void add_pandas_index_names(std::vector<std::string> &names) const
  {
    auto str = get_pandas_index();
    if (str.length() != 0) {
      std::regex index_name_expr{R"(\"((?:\\.|[^\"])*)\")"};
      std::smatch sm;
      while (std::regex_search(str, sm, index_name_expr)) {
        if (sm.size() == 2) {  // 2 = whole match, first item
          if (std::find(names.begin(), names.end(), sm[1].str()) == names.end()) {
            std::regex esc_quote{R"(\\")"};
            names.emplace_back(std::move(std::regex_replace(sm[1].str(), esc_quote, R"(")")));
          }
        }
        str = sm.suffix();
      }
    }
  }

  struct row_group_info {
    size_type const index;
    size_t const start_row;  // TODO source index
    size_type const source_index;
    row_group_info(size_type index, size_t start_row, size_type source_index)
      : index(index), start_row(start_row), source_index(source_index)
    {
    }
  };

  /**
   * @brief Filters and reduces down to a selection of row groups
   *
   * @param row_groups Lists of row group to reads, one per source
   * @param row_start Starting row of the selection
   * @param row_count Total number of rows selected
   *
   * @return List of row group indexes and its starting row
   */
  auto select_row_groups(std::vector<std::vector<size_type>> const &row_groups,
                         size_type &row_start,
                         size_type &row_count) const
  {
    if (!row_groups.empty()) {
      std::vector<row_group_info> selection;
      CUDF_EXPECTS(row_groups.size() == per_file_metadata.size(),
                   "Must specify row groups for each source");

      row_count = 0;
      for (size_t src_idx = 0; src_idx < row_groups.size(); ++src_idx) {
        for (auto const &rowgroup_idx : row_groups[src_idx]) {
          CUDF_EXPECTS(
            rowgroup_idx >= 0 &&
              rowgroup_idx < static_cast<size_type>(per_file_metadata[src_idx].row_groups.size()),
            "Invalid rowgroup index");
          selection.emplace_back(rowgroup_idx, row_count, src_idx);
          row_count += get_row_group(rowgroup_idx, src_idx).num_rows;
        }
      }
      return selection;
    }

    row_start = std::max(row_start, 0);
    if (row_count < 0) {
      row_count = static_cast<size_type>(
        std::min<int64_t>(get_num_rows(), std::numeric_limits<size_type>::max()));
    }
    CUDF_EXPECTS(row_count >= 0, "Invalid row count");
    CUDF_EXPECTS(row_start <= get_num_rows(), "Invalid row start");

    std::vector<row_group_info> selection;
    size_type count = 0;
    for (size_t src_idx = 0; src_idx < per_file_metadata.size(); ++src_idx) {
      for (size_t rg_idx = 0; rg_idx < per_file_metadata[src_idx].row_groups.size(); ++rg_idx) {
        auto const chunk_start_row = count;
        count += get_row_group(rg_idx, src_idx).num_rows;
        if (count > row_start || count == 0) {
          selection.emplace_back(rg_idx, chunk_start_row, src_idx);
        }
        if (count >= row_start + row_count) { break; }
      }
    }
    return selection;
  }

  /**
   * @brief Filters and reduces down to a selection of columns
   *
   * @param use_names List of column names to select
   * @param include_index Whether to always include the PANDAS index column(s)
   *
   * @return List of column names
   */
  auto select_columns(std::vector<std::string> use_names, bool include_index) const
  {
    std::vector<std::pair<int, std::string>> selection;
    if (use_names.empty()) {
      // No columns specified; include all in the dataset
      for (const auto &name : column_names) { selection.emplace_back(selection.size(), name); }
    } else {
      // Load subset of columns; include PANDAS index unless excluded
      if (include_index) { add_pandas_index_names(use_names); }
      for (const auto &use_name : use_names) {
        for (size_t i = 0; i < column_names.size(); ++i) {
          if (column_names[i] == use_name) {
            selection.emplace_back(i, column_names[i]);
            break;
          }
        }
      }
    }

    return selection;
  }
};

void reader::impl::read_column_chunks(
  std::vector<rmm::device_buffer> &page_data,
  hostdevice_vector<gpu::ColumnChunkDesc> &chunks,  // TODO const?
  size_t begin_chunk,
  size_t end_chunk,
  const std::vector<size_t> &column_chunk_offsets,
  std::vector<size_type> const &chunk_source_map,
  cudaStream_t stream)
{
  // Transfer chunk data, coalescing adjacent chunks
  for (size_t chunk = begin_chunk; chunk < end_chunk;) {
    const size_t io_offset   = column_chunk_offsets[chunk];
    size_t io_size           = chunks[chunk].compressed_size;
    size_t next_chunk        = chunk + 1;
    const bool is_compressed = (chunks[chunk].codec != parquet::Compression::UNCOMPRESSED);
    while (next_chunk < end_chunk) {
      const size_t next_offset = column_chunk_offsets[next_chunk];
      const bool is_next_compressed =
        (chunks[next_chunk].codec != parquet::Compression::UNCOMPRESSED);
      if (next_offset != io_offset + io_size || is_next_compressed != is_compressed) {
        // Can't merge if not contiguous or mixing compressed and uncompressed
        // Not coalescing uncompressed with compressed chunks is so that compressed buffers can be
        // freed earlier (immediately after decompression stage) to limit peak memory requirements
        break;
      }
      io_size += chunks[next_chunk].compressed_size;
      next_chunk++;
    }
    if (io_size != 0) {
      auto buffer         = _sources[chunk_source_map[chunk]]->host_read(io_offset, io_size);
      page_data[chunk]    = rmm::device_buffer(buffer->data(), buffer->size(), stream);
      uint8_t *d_compdata = reinterpret_cast<uint8_t *>(page_data[chunk].data());
      do {
        chunks[chunk].compressed_data = d_compdata;
        d_compdata += chunks[chunk].compressed_size;
      } while (++chunk != next_chunk);
    } else {
      chunk = next_chunk;
    }
  }
}

size_t reader::impl::count_page_headers(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                        cudaStream_t stream)
{
  size_t total_pages = 0;

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream));
  CUDA_TRY(cudaMemcpyAsync(
    chunks.host_ptr(), chunks.device_ptr(), chunks.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  for (size_t c = 0; c < chunks.size(); c++) {
    total_pages += chunks[c].num_data_pages + chunks[c].num_dict_pages;
  }

  return total_pages;
}

void reader::impl::decode_page_headers(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                       hostdevice_vector<gpu::PageInfo> &pages,
                                       cudaStream_t stream)
{
  // IMPORTANT : if you change this relationship, please update preprocess_nested_columns to
  // reflect this.
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    chunks[c].max_num_pages = chunks[c].num_data_pages + chunks[c].num_dict_pages;
    chunks[c].page_info     = pages.device_ptr(page_count);
    if(chunks[c].terminator){
      printf("Terminator : %lu %lu\n", c, (uint64_t)chunks[c].page_info);
    }
    page_count += chunks[c].max_num_pages;
  }

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  CUDA_TRY(gpu::DecodePageHeaders(chunks.device_ptr(), chunks.size(), stream));
  CUDA_TRY(cudaMemcpyAsync(
    pages.host_ptr(), pages.device_ptr(), pages.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));
}

rmm::device_buffer reader::impl::decompress_page_data(
  hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
  hostdevice_vector<gpu::PageInfo> &pages,
  cudaStream_t stream)
{
  auto for_each_codec_page = [&](parquet::Compression codec, const std::function<void(size_t)> &f) {
    for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
      const auto page_stride = chunks[c].max_num_pages;
      if (chunks[c].codec == codec) {
        for (int k = 0; k < page_stride; k++) { f(page_count + k); }
      }
      page_count += page_stride;
    }
  };

  // Brotli scratch memory for decompressing
  rmm::device_vector<uint8_t> debrotli_scratch;

  // Count the exact number of compressed pages
  size_t num_comp_pages    = 0;
  size_t total_decomp_size = 0;
  std::array<std::pair<parquet::Compression, size_t>, 3> codecs{std::make_pair(parquet::GZIP, 0),
                                                                std::make_pair(parquet::SNAPPY, 0),
                                                                std::make_pair(parquet::BROTLI, 0)};

  for (auto &codec : codecs) {
    for_each_codec_page(codec.first, [&](size_t page) {
      total_decomp_size += pages[page].uncompressed_page_size;
      codec.second++;
      num_comp_pages++;
    });
    if (codec.first == parquet::BROTLI && codec.second > 0) {
      debrotli_scratch.resize(get_gpu_debrotli_scratch_size(codec.second));
    }
  }

  // Dispatch batches of pages to decompress for each codec
  rmm::device_buffer decomp_pages(total_decomp_size, stream);
  hostdevice_vector<gpu_inflate_input_s> inflate_in(0, num_comp_pages, stream);
  hostdevice_vector<gpu_inflate_status_s> inflate_out(0, num_comp_pages, stream);

  size_t decomp_offset = 0;
  int32_t argc         = 0;
  for (const auto &codec : codecs) {
    if (codec.second > 0) {
      int32_t start_pos = argc;

      for_each_codec_page(codec.first, [&](size_t page) {
        auto dst_base              = static_cast<uint8_t *>(decomp_pages.data());
        inflate_in[argc].srcDevice = pages[page].page_data;
        inflate_in[argc].srcSize   = pages[page].compressed_page_size;
        inflate_in[argc].dstDevice = dst_base + decomp_offset;
        inflate_in[argc].dstSize   = pages[page].uncompressed_page_size;

        inflate_out[argc].bytes_written = 0;
        inflate_out[argc].status        = static_cast<uint32_t>(-1000);
        inflate_out[argc].reserved      = 0;

        pages[page].page_data = (uint8_t *)inflate_in[argc].dstDevice;
        decomp_offset += inflate_in[argc].dstSize;
        argc++;
      });

      CUDA_TRY(cudaMemcpyAsync(inflate_in.device_ptr(start_pos),
                               inflate_in.host_ptr(start_pos),
                               sizeof(decltype(inflate_in)::value_type) * (argc - start_pos),
                               cudaMemcpyHostToDevice,
                               stream));
      CUDA_TRY(cudaMemcpyAsync(inflate_out.device_ptr(start_pos),
                               inflate_out.host_ptr(start_pos),
                               sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
                               cudaMemcpyHostToDevice,
                               stream));
      switch (codec.first) {
        case parquet::GZIP:
          CUDA_TRY(gpuinflate(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos,
                              1,
                              stream))
          break;
        case parquet::SNAPPY:
          CUDA_TRY(gpu_unsnap(inflate_in.device_ptr(start_pos),
                              inflate_out.device_ptr(start_pos),
                              argc - start_pos,
                              stream));
          break;
        case parquet::BROTLI:
          CUDA_TRY(gpu_debrotli(inflate_in.device_ptr(start_pos),
                                inflate_out.device_ptr(start_pos),
                                debrotli_scratch.data().get(),
                                debrotli_scratch.size(),
                                argc - start_pos,
                                stream));
          break;
        default: CUDF_EXPECTS(false, "Unexpected decompression dispatch"); break;
      }
      CUDA_TRY(cudaMemcpyAsync(inflate_out.host_ptr(start_pos),
                               inflate_out.device_ptr(start_pos),
                               sizeof(decltype(inflate_out)::value_type) * (argc - start_pos),
                               cudaMemcpyDeviceToHost,
                               stream));
    }
  }
  CUDA_TRY(cudaStreamSynchronize(stream));

  // Update the page information in device memory with the updated value of
  // page_data; it now points to the uncompressed data buffer
  CUDA_TRY(cudaMemcpyAsync(
    pages.device_ptr(), pages.host_ptr(), pages.memory_size(), cudaMemcpyHostToDevice, stream));

  return decomp_pages;
}

void reader::impl::allocate_nesting_info(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                         hostdevice_vector<gpu::PageInfo> &pages,
                                         hostdevice_vector<gpu::PageNestingInfo> &page_nesting_info,                                         
                                         std::vector<std::vector<std::pair<int, bool>>>& col_nesting_info,
                                         int num_columns,
                                         cudaStream_t stream)
{
  // resize col_nesting_info
  col_nesting_info.resize(num_columns);  

  // compute total # of page_nesting infos needed and allocate space. doing this in one
  // buffer to keep it to a single gpu allocation
  int total_page_nesting_infos = 0;
  for(size_t idx=0; idx<chunks.size(); idx++){
    int col_index = chunks[idx].col_index;
    // the leaf schema represents the bottom of the nested hierarchy
    auto &leaf_schema = _metadata->get_column_leaf_schema(col_index);
    int per_page_nesting_info_size = leaf_schema.max_definition_level + 1;
    // IMPORTANT : see description of ColumnNestingInfo struct for information on how this is
    // accessed    
    //printf("PAGES : %lu %d %d %d\n", idx, per_page_nesting_info_size, chunks[idx].num_data_pages, total_page_nesting_infos);
    total_page_nesting_infos += (per_page_nesting_info_size * chunks[idx].num_data_pages);
  }
  //printf("TPNI : %d\n", total_page_nesting_infos);
  hostdevice_vector<gpu::PageNestingInfo> pni(total_page_nesting_infos);
  page_nesting_info = std::move(pni);

  // retrieve from the gpu so we can update
  pages.device_to_host(stream);

  // update pointers in the PageInfos
  int target_page_index = 0;
  int src_info_index = 0;
  for(size_t idx=0; idx<chunks.size(); idx++){
    int col_index = chunks[idx].col_index;
    auto &leaf_schema = _metadata->get_column_leaf_schema(col_index);
    int per_page_nesting_info_size = leaf_schema.max_definition_level + 1;
    
    // skip my dict pages
    target_page_index += chunks[idx].num_dict_pages;
    for(int p_idx=0; p_idx<chunks[idx].num_data_pages; p_idx++){      
      pages[target_page_index + p_idx].nesting = page_nesting_info.device_ptr() + src_info_index;
      pages[target_page_index + p_idx].max_nesting_depth = per_page_nesting_info_size;
      //printf("Page %d : src_info_index(%d), nesting ptr %lu\n", target_page_index + p_idx, src_info_index, (uint64_t)pages[target_page_index + p_idx].nesting);
      src_info_index += per_page_nesting_info_size;
    }    
    target_page_index += chunks[idx].num_data_pages;
  }

  // copy back to the gpu
  pages.host_to_device(stream);  

  // fill in 
  int nesting_info_index = 0;
  for (size_t idx=0; idx<chunks.size(); idx++) {
    int col_index = chunks[idx].col_index;

    // the leaf schema represents the bottom of the nested hierarchy
    auto &leaf_schema = _metadata->get_column_leaf_schema(col_index);
    // real depth of the output cudf column hiearchy (1 == no nesting, 2 == 1 level, etc)
    int max_depth = _metadata->get_nesting_depth(col_index);

    // # of nesting infos stored per page for this column
    size_t per_page_nesting_info_size = leaf_schema.max_definition_level + 1;

    col_nesting_info[col_index].resize(max_depth);

    // fill in host-side nesting info
    int schema_idx     = _metadata->get_column_leaf_schema_index(col_index);
    auto cur_schema    = _metadata->get_schema(schema_idx);
    int output_col_idx = max_depth - 1;
    while (schema_idx > 0) {
      // repetition type for this level
      FieldRepetitionType repetition_type = cur_schema.repetition_type;

      int d = cur_schema.max_definition_level;

      // set nullability on the column 
      if(repetition_type != REPEATED){
        col_nesting_info[col_index][output_col_idx].second = repetition_type == OPTIONAL ? true : false;
      }

      // initialize each page within the chunk
      for(int p_idx=0; p_idx<chunks[idx].num_data_pages; p_idx++){
        gpu::PageNestingInfo *pni = &page_nesting_info[nesting_info_index + (p_idx * per_page_nesting_info_size)];
        int input_index = d;
        int output_index = output_col_idx;

        // values indexed by definition level
        if(p_idx == 0){
          //printf("Remap d:%d -> %d\n", d, output_col_idx);
        }
        pni[input_index].d_remap = output_col_idx;

        // REPEATED fields are not "real" output cudf columns. they just represent a level of nesting.
        if (repetition_type != REPEATED) {
          // values indexed by output column index
          pni[output_index].o_max_def_level = d;
          pni[output_index].o_size          = 0;
          pni[output_index].o_null_count    = 0;
          // definition 0 always remaps to column 0.  
          if(output_index == 0){
            pni[output_index].d_remap = 0;
          }
        }
      }

      // move up the hierarchy

      // if this was a REPEATED field, it represents a level of nesting, so
      // move up the output column
      if(repetition_type == REPEATED){        
        output_col_idx--;
      }
      schema_idx = cur_schema.parent_idx;
      cur_schema = _metadata->get_schema(schema_idx);
    }

    nesting_info_index += (per_page_nesting_info_size * chunks[idx].num_data_pages);
  }  

  // copy nesting info to the device
  page_nesting_info.host_to_device(stream);

  // we have also updated the pages, so copy/update them on the device
  pages.host_to_device(stream);
}


void reader::impl::preprocess_nested_columns(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                             hostdevice_vector<gpu::PageInfo> &pages,
                                             hostdevice_vector<gpu::PageNestingInfo> &page_nesting_info,
                                             std::vector<std::vector<std::pair<int, bool>>>& nested_sizes,
                                             size_t min_row,
                                             size_t total_rows,
                                             cudaStream_t stream)
{  
  // preprocess per-nesting level sizes by page
  printf("PREPROCESS START %lu, %lu\n", min_row, total_rows);
  CUDA_TRY(gpu::PreprocessColumnData(pages,
                                     chunks,
                                     nested_sizes,
                                     total_rows,                                     
                                     min_row,
                                     stream));

  CUDA_TRY(cudaStreamSynchronize(stream));
  printf("PREPROCESS END\n");   
}

void reader::impl::decode_page_data(hostdevice_vector<gpu::ColumnChunkDesc> &chunks,
                                    hostdevice_vector<gpu::PageInfo> &pages,
                                    size_t min_row,
                                    size_t total_rows,                                    
                                    std::vector<column_buffer> &out_buffers,
                                    bool has_nesting,
                                    cudaStream_t stream)
{
  printf("DECODE START\n");

  auto is_dict_chunk = [](const gpu::ColumnChunkDesc &chunk) {
    return (chunk.data_type & 0x7) == BYTE_ARRAY && chunk.num_dict_pages > 0;
  };

  // Count the number of string dictionary entries
  // NOTE: Assumes first page in the chunk is always the dictionary page
  size_t total_str_dict_indexes = 0;
  for (size_t c = 0, page_count = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) { total_str_dict_indexes += pages[page_count].num_values; }
    page_count += chunks[c].max_num_pages;
  }

  // Build index for string dictionaries since they can't be indexed
  // directly due to variable-sized elements
  rmm::device_vector<gpu::nvstrdesc_s> str_dict_index;
  if (total_str_dict_indexes > 0) { str_dict_index.resize(total_str_dict_indexes); }

  // Update chunks with pointers to column data. 
  // Note : each output column may have multiple "chunks", each representing
  // an individual row group.  all of them store the same output buffers, and the
  // pages that make up an individual chunk index them in an absolute way.
  for (size_t c = 0, page_count = 0, str_ofs = 0; c < chunks.size(); c++) {
    if (is_dict_chunk(chunks[c])) {
      chunks[c].str_dict_index = str_dict_index.data().get() + str_ofs;
      str_ofs += pages[page_count].num_values;
    }

    int output_depth = chunks[c].max_level[gpu::level_type::REPETITION];

    // setup base pointers. need to do this better
    size_type buf_bytes = sizeof(void *) * (output_depth + 1);
    cudaMalloc(&chunks[c].valid_map_base, buf_bytes);
    cudaMalloc(&chunks[c].column_data_base, buf_bytes);
    std::vector<uint32_t *> valids(output_depth + 1);
    std::vector<void *> data(output_depth + 1);
    column_buffer *buf = &out_buffers[chunks[c].col_index];
    for (int idx = 0; idx <= output_depth; idx++) {
      data[idx]   = buf->data();
      valids[idx] = buf->null_mask();
      if (idx < output_depth) {
        CUDF_EXPECTS(buf->children.size() > 0, "Encountered a malformed column_buffer");
        buf = &buf->children[0];
      }
    }
    cudaMemcpy(chunks[c].valid_map_base, valids.data(), buf_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(chunks[c].column_data_base, data.data(), buf_bytes, cudaMemcpyHostToDevice);

    // column_data_base will always point to leaf data, even for nested types.
    page_count += chunks[c].max_num_pages;
  }

  CUDA_TRY(cudaMemcpyAsync(
    chunks.device_ptr(), chunks.host_ptr(), chunks.memory_size(), cudaMemcpyHostToDevice, stream));
  if (total_str_dict_indexes > 0) {
    CUDA_TRY(gpu::BuildStringDictionaryIndex(chunks.device_ptr(), chunks.size(), stream));
  }

  printf("DecodePageData : pages size : %lu, chunks size : %lu, total rows : %lu, min row : %lu\n",
         pages.size(),
         chunks.size(),
         total_rows,
         min_row);
  CUDA_TRY(gpu::DecodePageData(pages.device_ptr(),
                               pages.size(),
                               chunks.device_ptr(),
                               chunks.size(),
                               total_rows,
                               min_row,
                               has_nesting,
                               stream));
  CUDA_TRY(cudaMemcpyAsync(
    pages.host_ptr(), pages.device_ptr(), pages.memory_size(), cudaMemcpyDeviceToHost, stream));
  CUDA_TRY(cudaStreamSynchronize(stream));

  /*
  for (size_t i = 0; i < pages.size(); i++) {
    if (pages[i].num_rows > 0) {
      const size_t c = pages[i].chunk_idx;
      if (c < chunks.size()) {
        out_buffers[chunk_col_map[c]].null_count() += pages[i].num_rows - pages[i].valid_count;
      }
    }
  }
  */
  printf("DECODE END\n");
}

reader::impl::impl(std::vector<std::unique_ptr<datasource>> &&sources,
                   reader_options const &options,
                   rmm::mr::device_memory_resource *mr)
  : _sources(std::move(sources)), _mr(mr)
{
  // Open and parse the source dataset metadata
  _metadata = std::make_unique<aggregate_metadata>(_sources);

  // Select only columns required by the options
  _selected_columns = _metadata->select_columns(options.columns, options.use_pandas_metadata);

  // Override output timestamp resolution if requested
  if (options.timestamp_type.id() != type_id::EMPTY) { _timestamp_type = options.timestamp_type; }

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.strings_to_categorical;
}

table_with_metadata reader::impl::read(size_type skip_rows,
                                       size_type num_rows,
                                       std::vector<std::vector<size_type>> const &row_group_list,
                                       cudaStream_t stream)
{
  // Select only row groups required
  const auto selected_row_groups =
    _metadata->select_row_groups(row_group_list, skip_rows, num_rows);

  // Get a list of column data types
  std::vector<data_type> column_types;
  if (_metadata->get_num_row_groups() != 0) {
    for (const auto &col : _selected_columns) {
      auto &col_schema = _metadata->get_column_schema(col.first);
      auto col_type    = to_type_id(col_schema, _strings_to_categorical, _timestamp_type.id());
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");
      column_types.emplace_back(col_type);
    }
  }

  std::vector<std::unique_ptr<column>> out_columns;
  printf("Selected %lu columns\n", column_types.size());
  out_columns.reserve(column_types.size());

  if (selected_row_groups.size() != 0 && column_types.size() != 0) {
    // Descriptors for all the chunks that make up the selected columns
    const auto num_columns = _selected_columns.size();
    const auto num_chunks  = selected_row_groups.size() * num_columns;
    hostdevice_vector<gpu::ColumnChunkDesc> chunks(0, num_chunks, stream);
  
    // Association between each column chunk and its source
    std::vector<size_type> chunk_source_map(num_chunks);

    // Tracker for eventually deallocating compressed and uncompressed data
    std::vector<rmm::device_buffer> page_data(num_chunks);

    // Keep track of column chunk file offsets
    std::vector<size_t> column_chunk_offsets(num_chunks);

    // information needed allocate columns (including potential nesting)
    bool has_nesting = false;

    // Initialize column chunk information
    size_t total_decompressed_size = 0;
    auto remaining_rows            = num_rows;
    for (const auto &rg : selected_row_groups) {
      const auto &row_group       = _metadata->get_row_group(rg.index, rg.source_index);
      auto const row_group_start  = rg.start_row;
      auto const row_group_source = rg.source_index;
      auto const row_group_rows   = std::min<int>(remaining_rows, row_group.num_rows);
      auto const io_chunk_idx     = chunks.size();
      bool terminating_rg = (remaining_rows - row_group.num_rows) == 0 ? true : false;

      for (size_t i = 0; i < num_columns; ++i) {
        auto col       = _selected_columns[i];
        auto &col_meta = row_group.columns[col.first].meta_data;
        
        // the root schema (which in the case of nested types is different from the leaf schema).
        // the # of rows in the row group is relative to the root
        // auto const &col_schema = _metadata->get_schema(row_group.columns[col.first].schema_idx);
        auto &root_schema = _metadata->get_column_schema(col.first);
        // the leaf schema represents the -values- encoded in the data, which in the case
        // of nested types, is different from the # of rows
        auto &leaf_schema = _metadata->get_column_leaf_schema(col.first);

        // this file contains nesting and will require a preprocess
        if(_metadata->get_nesting_depth(col.first) > 1){
          has_nesting = true;
        }

        // Spec requires each row group to contain exactly one chunk for every
        // column. If there are too many or too few, continue with best effort
        if (col.second != name_from_path(col_meta.path_in_schema)) {
          std::cerr << "Detected mismatched column chunk" << std::endl;
          continue;
        }
        if (chunks.size() >= chunks.max_size()) {
          std::cerr << "Detected too many column chunks" << std::endl;
          continue;
        }

        int32_t type_width;
        int32_t clock_rate;
        int8_t converted_type;
        std::tie(type_width, clock_rate, converted_type) =
          conversion_info(column_types[i].id(),
                          _timestamp_type.id(),
                          leaf_schema.type,
                          leaf_schema.converted_type,
                          leaf_schema.type_length);

        column_chunk_offsets[chunks.size()] =
          (col_meta.dictionary_page_offset != 0)
            ? std::min(col_meta.data_page_offset, col_meta.dictionary_page_offset)
            : col_meta.data_page_offset;
        
        // printf("RG : %d %lu %d %lu\n", rg_index, row_group_start, row_group_rows, col_meta.num_values);
        chunks.insert(gpu::ColumnChunkDesc(col_meta.total_compressed_size,
                                           nullptr,
                                           col_meta.num_values,
                                           leaf_schema.type,
                                           type_width,
                                           row_group_start,
                                           row_group_rows,
                                           leaf_schema.max_definition_level,
                                           leaf_schema.max_repetition_level,
                                           required_bits(leaf_schema.max_definition_level),
                                           required_bits(leaf_schema.max_repetition_level),
                                           col_meta.codec,
                                           converted_type,
                                           leaf_schema.decimal_scale,
                                           clock_rate,
                                           i,
                                           terminating_rg));

        // Map each column chunk to its column index and its source index
        chunk_source_map[chunks.size() - 1] = row_group_source;

        if (col_meta.codec != Compression::UNCOMPRESSED) {
          total_decompressed_size += col_meta.total_uncompressed_size;
        }
      }
      // Read compressed chunk data to device memory
      read_column_chunks(page_data,
                         chunks,
                         io_chunk_idx,
                         chunks.size(),
                         column_chunk_offsets,
                         chunk_source_map,
                         stream);

      remaining_rows -= row_group.num_rows;
    }
    assert(remaining_rows <= 0);

    // Process dataset chunk pages into output columns
    const auto total_pages = count_page_headers(chunks, stream);
    if (total_pages > 0) {            
      hostdevice_vector<gpu::PageInfo> pages(total_pages, total_pages, stream);      
      rmm::device_buffer decomp_page_data;
            
      // decoding of column/page information
      decode_page_headers(chunks, pages, stream);
      if (total_decompressed_size > 0) {
        decomp_page_data = decompress_page_data(chunks, pages, stream);
        // Free compressed data
        for (size_t c = 0; c < chunks.size(); c++) {
          if (chunks[c].codec != parquet::Compression::UNCOMPRESSED && page_data[c].size() != 0) {
            page_data[c].resize(0);
            page_data[c].shrink_to_fit();
          }
        }
      }

      // nesting information (sizes, etc) stored -per page-
      hostdevice_vector<gpu::PageNestingInfo> page_nesting_info;
      // nesting information at the column level. 
      // - total column size per nesting level
      // - nullability per nesting level
      std::vector<std::vector<std::pair<int, bool>>> col_nesting_info;

      // even for flat schemas, we allocate 1 level of "nesting" info      
      allocate_nesting_info(chunks, pages, page_nesting_info, col_nesting_info, num_columns, stream);

      // for nested schemas, we have to do some further preprocessing to determine:
      // - real column output sizes per level of nesting (in a flat schema, there's only 1 level of
      //   nesting and it's size is the row count)
      //
      // - output buffer offset values per-page, per nesting-level for the purposes of decoding.      
      if(has_nesting){      
        preprocess_nested_columns(chunks, pages, page_nesting_info, col_nesting_info, skip_rows, num_rows, stream);
      }

      std::vector<column_buffer> out_buffers;
      out_buffers.reserve(column_types.size());
      for (size_t i = 0; i < column_types.size(); ++i) {
        auto col          = _selected_columns[i];
        auto &leaf_schema = _metadata->get_column_leaf_schema(col.first);

        int output_depth = leaf_schema.max_repetition_level + 1;

        // nested schemas : sizes and nullability come from preprocess step
        if (output_depth > 1) {
          // the root buffer
          out_buffers.emplace_back(column_buffer{column_types[i],
                                                 col_nesting_info[i][0].first,
                                                 col_nesting_info[i][0].second,
                                                 stream,
                                                 _mr});
          column_buffer *col = &out_buffers[out_buffers.size() - 1];
          // nested buffers
          for (int idx = 1; idx < output_depth - 1; idx++) {
            // note : all levels in a list column besides the leaf are offsets, so their length is
            // always +1
            col->children.push_back(column_buffer{column_types[i],
                                                  col_nesting_info[i][idx].first,
                                                  col_nesting_info[i][idx].second,
                                                  stream,
                                                  _mr});
            col = &col->children[0];
          }
          // leaf buffer - plain data type. int, string, etc
          col->children.push_back(column_buffer{
            data_type{to_type_id(leaf_schema, _strings_to_categorical, _timestamp_type.id())},
            col_nesting_info[i][output_depth - 1].first,
            col_nesting_info[i][output_depth - 1].second,
            stream,
            _mr});
        }
        // flat schemas can infer sizes directly from # of rows
        else {
          // note : num_rows == # values for non-nested types
          bool is_nullable = leaf_schema.max_definition_level != 0;
          out_buffers.emplace_back(column_buffer{column_types[i], num_rows, is_nullable, stream, _mr});
        }
      }

      // decoding of column data itself
      decode_page_data(
        chunks, pages, skip_rows, num_rows, out_buffers, has_nesting, stream);
    
      for (size_t i = 0; i < column_types.size(); ++i) {
        /*
        // retrieve validity counts
        nested_column_info[i].device_to_host(stream);

        auto col          = _selected_columns[i];
        auto &leaf_schema = _metadata->get_column_leaf_schema(col.first);
        int nesting_depth = _metadata->get_nesting_depth(col.first);

        uint8_t buf[512];
        column_buffer *cb = &out_buffers[i];
        for (int idx = 0; idx < nesting_depth; idx++) {
          cb->_null_count = nested_column_info[i][idx].o_null_count;
          cb              = &cb->children[0];
        }
        */
        out_columns.emplace_back(make_column(out_buffers[i], stream, _mr));        
      }
    }
  }

//done:
  PRINTF("STAGE 3\n");

  // Create empty columns as needed
  for (size_t i = out_columns.size(); i < column_types.size(); ++i) {
    out_columns.emplace_back(make_empty_column(column_types[i]));
  }

  table_metadata out_metadata;
  // Return column names (must match order of returned columns)
  out_metadata.column_names.resize(_selected_columns.size());
  for (size_t i = 0; i < _selected_columns.size(); i++) {
    out_metadata.column_names[i] = _selected_columns[i].second;
  }
  // Return user metadata
  out_metadata.user_data = _metadata->get_key_value_metadata();

  return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
}

// Forward to implementation
reader::reader(std::vector<std::string> const &filepaths,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(datasource::create(filepaths), options, mr))
{
}

// Forward to implementation
reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>> &&sources,
               reader_options const &options,
               rmm::mr::device_memory_resource *mr)
  : _impl(std::make_unique<impl>(std::move(sources), options, mr))
{
}

// Destructor within this translation unit
reader::~reader() = default;

// Forward to implementation
table_with_metadata reader::read_all(cudaStream_t stream) { return _impl->read(0, -1, {}, stream); }

// Forward to implementation
table_with_metadata reader::read_row_groups(std::vector<std::vector<size_type>> const &row_groups,
                                            cudaStream_t stream)
{
  return _impl->read(0, -1, row_groups, stream);
}

// Forward to implementation
table_with_metadata reader::read_rows(size_type skip_rows, size_type num_rows, cudaStream_t stream)
{
  return _impl->read(skip_rows, (num_rows != 0) ? num_rows : -1, {}, stream);
}

}  // namespace parquet
}  // namespace detail
}  // namespace io
}  // namespace cudf
