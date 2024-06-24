/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.
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

#include <cudf_test/base_fixture.hpp>
#include <cudf_test/column_utilities.hpp>
#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>
#include <cudf_test/testing_main.hpp>
#include <cudf_test/type_lists.hpp>

#include <cudf/detail/iterator.cuh>
#include <cudf/hashing.hpp>

constexpr cudf::test::debug_output_level verbosity{cudf::test::debug_output_level::ALL_ERRORS};

class HiveHashTest : public cudf::test::BaseFixture {};

TEST_F(HiveHashTest, INT32)
{
  // TODO
}

TEST_F(HiveHashTest, INT64)
{
  // TODO
}

TEST_F(HiveHashTest, COLUMNS)
{
  cudf::test::strings_column_wrapper const strings_col(
    {"",
     "The quick brown fox",
     "jumps over the lazy dog.",
     "All work and no play makes Jack a dull boy",
     R"(!"#$%&'()*+,-./0123456789:;<=>?@[\]^_`{|}~)"});

  using limits = std::numeric_limits<int32_t>;
  cudf::test::fixed_width_column_wrapper<int32_t> const ints_col(
    {0, 100, -100, limits::min(), limits::max()});

  // Different truth values should be equal
  cudf::test::fixed_width_column_wrapper<bool> const bools_col1({0, 1, 1, 1, 0});
  cudf::test::fixed_width_column_wrapper<bool> const bools_col2({0, 1, 2, 255, 0});

  using ts = cudf::timestamp_s;
  cudf::test::fixed_width_column_wrapper<ts, ts::duration> const secs_col(
    {ts::duration::zero(),
     static_cast<ts::duration>(100),
     static_cast<ts::duration>(-100),
     ts::duration::min(),
     ts::duration::max()});

  auto const input1 = cudf::table_view({strings_col, ints_col, bools_col1, secs_col});
  auto const input2 = cudf::table_view({strings_col, ints_col, bools_col2, secs_col});

  auto const output1 = cudf::hashing::hive_hash(input1);
  auto const output2 = cudf::hashing::hive_hash(input2);

  EXPECT_EQ(input1.num_rows(), output1->size());
  CUDF_TEST_EXPECT_COLUMNS_EQUAL(output1->view(), output2->view());
}

CUDF_TEST_PROGRAM_MAIN()
