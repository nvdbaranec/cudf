/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

#include <cudf/types.hpp>

/**
 * @file list_view.cuh
 * @brief Class definition for cudf::list_view.
 */

namespace cudf {

class list_view {
public:

  /**
   * @brief Default constructor represents an empty list.
   */
  __host__ __device__ list_view(){}
  
  list_view(const list_view&) = default;
  list_view(list_view&&) = default;
  ~list_view() = default;
  list_view& operator=(const list_view&) = default;
  list_view& operator=(list_view&&) = default;
};

} // namespace cudf
