/*
 * Copyright (c) 2024-2024, NVIDIA CORPORATION.
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

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/types.hpp>

#include <cstddef>

namespace cudf::hashing::detail {

// hive hash implementation referring to apache spark HiveHasher.java
template <typename Key>
struct HiveHash {

  hash_value_type __device__ inline compute_bytes(std::byte const *data, cudf::size_type const n_bytes) const
  {
    hash_value_type res = 0;

    for (auto i = 0; i < n_bytes; i++)
    {
        res = res * 31 + data[i];
    }

    return res;
  }

  template <typename T>
  hash_value_type __device__ inline compute(T const& key)
  {
    return compute_bytes(reinterpret_cast<std::byte const*>(&key), sizeof(T));
  }

  [[nodiscard]] hash_value_type __device__ inline operator()(Key const& key) const
  {
    return compute(normalize_nans_and_zeros(key));
  }

};

template <>
hash_value_type __device__ inline HiveHash<int32_t>::operator()(int32_t const& key) const
{
  return key;
}

template <>
hash_value_type __device__ inline HiveHash<int64_t>::operator()(int64_t const& key) const
{
  return ((uint64_t)key >> 32)^key;
}

template <>
hash_value_type __device__ inline HiveHash<cudf::string_view>::operator()(cudf::string_view const& key) const
{
  auto const data = reinterpret_cast<std::byte const*>(key.data());
  auto const len  = key.size_bytes();
  return compute_bytes(data, len);
}

}  // namespace cudf::hashing::detail
