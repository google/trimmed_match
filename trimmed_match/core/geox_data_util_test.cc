/*
* Copyright 2020 Google LLC.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     https://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
* ============================================================================
*/
#include "trimmed_match/core/geox_data_util.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

namespace trimmedmatch {
namespace {
class GeoxDataUtilInternalTest : public ::testing::Test {
 protected:
  GeoxDataUtilInternalTest()
      : geox_data_({{3.0, 2.0}, {8.0, 5.0}, {10.0, 10.0}}),
        geox_data2_({{1.5391406, -1.7904224},
                     {1.3765249, -1.3941344},
                     {-1.0470811, -0.9608362},
                     {0.4050382, -0.1808448},
                     {-1.1829934, 0.2950698},
                     {-0.1920367, 0.3012471},
                     {-2.1449249, 0.5700090},
                     {0.2685824, 0.7528970},
                     {1.0362271, 0.7777585},
                     {-0.2633370, 1.1244019}}),
        geox_util_(geox_data_),
        geox_util2_(geox_data2_) {}

  const std::vector<GeoPairValues> geox_data_;
  const std::vector<GeoPairValues> geox_data2_;
  const GeoxDataUtil geox_util_;
  const GeoxDataUtil geox_util2_;
};

TEST(FindFirstJumpInDeltaCostFrom, SortedOrNot) {
  const std::vector<GeoPairValues> geox_data1 = {
      {0.0, 2.0}, {1.0, 1.0}, {2.0, 1.0}};
  const std::vector<GeoPairValues> geox_data2 = {
      {1.0, 1.0}, {2.0, 1.0 + 0.5 * kTieBreaker}, {0.0, 2.0}};

  EXPECT_EQ(FindFirstJumpInDeltaCostFrom(geox_data1, 0), 1);
  EXPECT_EQ(FindFirstJumpInDeltaCostFrom(geox_data1, 1), 3);
  EXPECT_EQ(FindFirstJumpInDeltaCostFrom(geox_data1, 3), 3);

  EXPECT_EQ(FindFirstJumpInDeltaCostFrom(geox_data2, 0), 1);
  EXPECT_EQ(FindFirstJumpInDeltaCostFrom(geox_data2, 1), 2);
  EXPECT_EQ(FindFirstJumpInDeltaCostFrom(geox_data2, 3), 3);
}

TEST(GeoxDataUtilTest, GeoXDataUtilWithDeltaCostTied) {
  const std::vector<GeoPairValues> geox_data = {
      {0.0, 2.0}, {1.0, 1.0}, {2.0, 1.0}};
  const std::vector<GeoPairValues> expected_geox_data = {
      {1.0, 1.0}, {2.0, 1.0 + 0.5 * kTieBreaker}, {0.0, 2.0}};
  GeoxDataUtil geox_util(geox_data);
  const std::vector<GeoPairValues> geo_paired_values =
      geox_util.ExtractGeoxData();

  for (int i = 0; i < geox_data.size(); ++i) {
    EXPECT_EQ(geo_paired_values[i].delta_response,
              expected_geox_data[i].delta_response);
    EXPECT_NEAR(geo_paired_values[i].delta_cost,
                expected_geox_data[i].delta_cost, kTieBreaker);
  }
}

TEST(GeoxDataUtilTest, GeoXDataUtilWithDeltaResponseTied) {
  const std::vector<GeoPairValues> geox_data = {{1, 4}, {1, -3}, {1, 1}};
  const GeoxDataUtil geox_util(geox_data);
  const std::vector<GeoPairValues> geo_paired_values =
      geox_util.ExtractGeoxData();
  // We sort the geox_data by the delta_cost first.
  const std::vector<GeoPairValues> expected_geox_data = {
      {/*delta_response=*/1, /*delta_cost=*/-3},
      {/*delta_response=*/1, /*delta_cost=*/1},
      {/*delta_response=*/1, /*delta_cost=*/4}};
  for (int i = 0; i < geox_data.size(); ++i) {
    EXPECT_EQ(expected_geox_data[i].delta_response,
              geo_paired_values[i].delta_response);
    EXPECT_EQ(expected_geox_data[i].delta_cost,
              geo_paired_values[i].delta_cost);
  }

  const std::vector<PairedDelta> paired_delta = geox_util.ExtractPairedDelta();
  // We generate 3 delta in total. Since all delta reponses are the same,
  // delta_ij in paired_delta are all 0. However, the order of paired_delta has
  // to be determined even if there is tie in delta_ij.
  const std::vector<PairedDelta> expected_paired_delta = {
      {/*i=*/2, /*j=*/1, /*delta_ij=*/0},
      {/*i=*/2, /*j=*/0, /*delta_ij=*/0},
      {/*i=*/1, /*j=*/0, /*delta_ij=*/0},
  };
  for (int n = 0; n < paired_delta.size(); ++n) {
    EXPECT_EQ(expected_paired_delta[n].i, paired_delta[n].i);
    EXPECT_EQ(expected_paired_delta[n].j, paired_delta[n].j);
    EXPECT_EQ(expected_paired_delta[n].delta, paired_delta[n].delta);
  }
}

TEST_F(GeoxDataUtilInternalTest, ExtractGeoxData) {
  const std::vector<GeoPairValues> result = geox_util_.ExtractGeoxData();

  for (int i = 0; i < geox_data_.size(); ++i) {
    EXPECT_EQ(geox_data_[i].delta_response, result[i].delta_response);
    EXPECT_EQ(geox_data_[i].delta_cost, result[i].delta_cost);
  }
}

TEST_F(GeoxDataUtilInternalTest, ExtractDeltaCost) {
  const std::vector<double> result = geox_util_.ExtractDeltaCost();

  for (int i = 0; i < geox_data_.size(); ++i) {
    EXPECT_EQ(geox_data_[i].delta_cost, result[i]);
  }
}

TEST_F(GeoxDataUtilInternalTest, CalculateResiduals) {
  const double iroas = 1.0;
  const std::vector<double> result = geox_util_.CalculateResiduals(iroas);

  ASSERT_EQ(geox_data_.size(), result.size());
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(geox_data_[i].delta_response - iroas * geox_data_[i].delta_cost,
                result[i], 1e-10);
  }
}

TEST_F(GeoxDataUtilInternalTest, CalculateEmpiricalIroas) {
  const double result = geox_util_.CalculateEmpiricalIroas();
  const std::vector<double> delta_response = geox_util_.ExtractDeltaResponse();
  const double sum_delta_response =
      std::accumulate(delta_response.begin(), delta_response.end(), 0.0);
  const std::vector<double> delta_cost = geox_util_.ExtractDeltaCost();
  const double sum_delta_cost =
      std::accumulate(delta_cost.begin(), delta_cost.end(), 0.0);

  EXPECT_NEAR(sum_delta_response, result * sum_delta_cost, 1e-10);
}

TEST_F(GeoxDataUtilInternalTest, FindAllZerosOfTrimmedMeanInvalidTrimRate) {
  EXPECT_DEATH(auto result = geox_util_.FindAllZerosOfTrimmedMean(0.6), "");
  EXPECT_DEATH(auto result = geox_util_.FindAllZerosOfTrimmedMean(-0.1), "");
}

TEST(FindAllZerosOfTrimmedMeanTest, FindAllZerosOfTrimmedMean0Root) {
  GeoxDataUtil geox_util({{1, -10}, {2, -1}, {3, 1}, {4, 10}});
  EXPECT_TRUE(geox_util.FindAllZerosOfTrimmedMean(0.25).empty());
}

TEST_F(GeoxDataUtilInternalTest, FindAllZerosOfTrimmedMean1Root) {
  const double expected =
      geox_data_[0].delta_response / geox_data_[0].delta_cost;

  const std::vector<double> result = geox_util_.FindAllZerosOfTrimmedMean(0.1);
  ASSERT_EQ(result.size(), 1);
  EXPECT_EQ(expected, result[0]);
}

TEST_F(GeoxDataUtilInternalTest, FindAllZerosOfTrimmedMean3Roots) {
  const double trim_rate = 0.1;
  const std::vector<double> expected = {-9.187, -0.488, 0.560};
  const std::vector<double> result =
      geox_util2_.FindAllZerosOfTrimmedMean(trim_rate);

  ASSERT_EQ(result.size(), expected.size());
  for (int i = 0; i < result.size(); ++i) {
    EXPECT_NEAR(expected[i], result[i], 1e-3);
  }

  for (int i = 0; i < result.size(); ++i) {
    std::vector<double> residuals = geox_util2_.CalculateResiduals(result[i]);
    std::sort(residuals.begin(), residuals.end());
    size_t num_trim = static_cast<size_t>(std::ceil(result.size() * trim_rate));
    const double result = std::accumulate(residuals.begin() + num_trim,
                                          residuals.end() - num_trim, 0.0);
    EXPECT_NEAR(0.0, result, 1e-10);
  }
}

TEST_F(GeoxDataUtilInternalTest, RangeFromStudentizedTrimmedMeanInvalidInput) {
  EXPECT_DEATH(
      auto result = geox_util_.RangeFromStudentizedTrimmedMean(-0.1, 1), "");
  EXPECT_DEATH(auto result = geox_util_.RangeFromStudentizedTrimmedMean(0.6, 1),
               "");
  EXPECT_DEATH(
      auto result = geox_util_.RangeFromStudentizedTrimmedMean(0.1, -1), "");
}

TEST_F(GeoxDataUtilInternalTest, RangeFromStudentizedTrimmedMeanNoTrim) {
  const std::pair<double, double> expected = {0.985, 2.180};
  const std::pair<double, double> result =
      geox_util_.RangeFromStudentizedTrimmedMean(0.0, 1.645);
  const std::pair<double, double> result2 =
      geox_util2_.RangeFromStudentizedTrimmedMean(0.0, 1.645);

  EXPECT_NEAR(expected.first, result.first, 0.001);
  EXPECT_NEAR(expected.second, result.second, 0.001);
  EXPECT_NEAR(-kBoundIroas, result2.first, 0.001);
  EXPECT_NEAR(kBoundIroas, result2.second, 0.001);
}

TEST_F(GeoxDataUtilInternalTest, RangeFromStudentizedTrimmedMeanWithTrim1) {
  std::vector<GeoPairValues> geox_data(geox_data_);
  geox_data.push_back({0.0, 0.0});
  geox_data.push_back({20.0, 20.0});
  GeoxDataUtil geox_util(geox_data);
  const std::pair<double, double> expected = {0.978, 9.400};
  const std::pair<double, double> result =
      geox_util.RangeFromStudentizedTrimmedMean(0.10, 1.645);

  EXPECT_NEAR(expected.first, result.first, 0.001);
  EXPECT_NEAR(expected.second, result.second, 0.001);
}

TEST_F(GeoxDataUtilInternalTest, RangeFromStudentizedTrimmedMeanWithTrim2) {
  const double threshold = 0.05;
  const double trim_rate = 0.1;
  const std::pair<double, double> expected = {-76.312, 0.898};
  const std::pair<double, double> result =
      geox_util2_.RangeFromStudentizedTrimmedMean(trim_rate, threshold);

  EXPECT_NEAR(expected.first, result.first, 0.001);
  EXPECT_NEAR(expected.second, result.second, 0.001);

  // Cross validation with StudentizedTrimmedMean().
  const std::vector<double> res1 =
      geox_util2_.CalculateResiduals(expected.first - 0.001);
  const std::vector<double> res2 =
      geox_util2_.CalculateResiduals(expected.second + 0.001);

  EXPECT_LE(threshold, std::abs(StudentizedTrimmedMean(res1, trim_rate)));
  EXPECT_LE(threshold, std::abs(StudentizedTrimmedMean(res2, trim_rate)));
}

}  // namespace
}  // namespace trimmedmatch
