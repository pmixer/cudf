/*
 * Copyright (c) 2024, NVIDIA CORPORATION.
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

#include <rmm/resource_ref.hpp>

#include <optional>

namespace cudf {

/**
 * @brief Set the rmm resource to be used for pinned memory allocations.
 *
 * @param mr The rmm resource to be used for pinned allocations
 * @return The previous resource that was in use
 */
rmm::host_device_async_resource_ref set_pinned_memory_resource(
  rmm::host_device_async_resource_ref mr);

/**
 * @brief Get the rmm resource being used for pinned memory allocations.
 *
 * @return The rmm resource used for pinned allocations
 */
rmm::host_device_async_resource_ref get_pinned_memory_resource();

/**
 * @brief Options to configure the default pinned memory resource
 */
struct pinned_mr_options {
  std::optional<size_t> pool_size;  ///< The size of the pool to use for the default pinned memory
                                    ///< resource. If not set, the default pool size is used.
};

/**
 * @brief Configure the size of the default pinned memory resource.
 *
 * @param opts Options to configure the default pinned memory resource
 * @return True if this call successfully configured the pinned memory resource, false if a
 * a resource was already configured.
 */
bool config_default_pinned_memory_resource(pinned_mr_options const& opts);

}  // namespace cudf
