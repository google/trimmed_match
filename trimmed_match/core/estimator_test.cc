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
#include "trimmed_match/core/estimator.h"

#include <cmath>

#include "gtest/gtest.h"

namespace trimmedmatch {
namespace {

void CompareReports(const Result& expected,
                    const absl::StatusOr<Result>& result,
                    const double epsilon = 1e-6) {
  if (result.ok()) {
    EXPECT_NEAR(expected.estimate, result->estimate, epsilon);
    EXPECT_NEAR(expected.std_error, result->std_error, epsilon);
    EXPECT_NEAR(expected.trim_rate, result->trim_rate, epsilon);
    EXPECT_NEAR(expected.conf_interval_low, result->conf_interval_low, epsilon);
    EXPECT_NEAR(expected.conf_interval_up, result->conf_interval_up, epsilon);
    EXPECT_EQ(expected.candidate_results.size(),
              result->candidate_results.size());

    for (size_t i = 0; i < result->candidate_results.size(); ++i) {
      EXPECT_NEAR(expected.candidate_results[i].iroas,
                  result->candidate_results[i].iroas, epsilon);
    }
  }
}

class EstimatorInternalTest : public ::testing::Test {
 protected:
  EstimatorInternalTest()
      : delta_response1_({1.0, 10.0, 3.0, 8.0}),
        delta_cost1_({1.0, 5.0, 2.0, 5.0}),
        trimmed_match1_(delta_response1_, delta_cost1_),
        trimmed_match2_(
            {552,   -710,  -961,   -1218,  -1293,  -1824, 1816,   2073,
             -2416, -2559, 2603,   3073,   3288,   3137,  4016,   3905,
             4032,  -4649, 5397,   -6054,  6451,   7735,  8507,   -9186,
             9700,  9987,  -11255, -13828, -14950, 20957, -38666, 43407},
            {89143,    -111084, -163240,  -201668, -213271, -311807,  313531,
             350041,   -425882, -439139,  431609,  509310,  554237,   548335,
             677060,   654552,  686768,   -812570, 940149,  -1019919, 1086827,
             1333913,  1458086, -1550027, 1676807, 1692128, -1927924, -2341819,
             -2533673, 3570161, -6614810, 7440206},
            0.30) {}

  const std::vector<double> delta_response1_;
  const std::vector<double> delta_cost1_;
  const TrimmedMatch trimmed_match1_;
  const TrimmedMatch trimmed_match2_;
};

TEST(TrimmedMatchInitialization, TrimmedMatchInvalidInput) {
  EXPECT_DEATH(auto result = TrimmedMatch({1, 2, 3}, {1, 2}), "");
}

TEST_F(EstimatorInternalTest, CalculateIroasEmptyRoot) {
  TrimmedMatch trimmed_match({1, 2, 3, 4}, {-10, -1, 1, 10});
  EXPECT_FALSE(trimmed_match.CalculateIroas(0.1).ok());
  EXPECT_EQ(
      trimmed_match.CalculateIroas(0.1).status(),
      absl::InternalError(
          "We could not find a root for the TM equation. One likely reason is "
          "that the incremental cost for the untrimmed geo pairs is 0."));
}

TEST_F(EstimatorInternalTest, InvalidInput) {
  EXPECT_FALSE(trimmed_match1_.CalculateIroas(-0.25).ok());
  EXPECT_EQ(trimmed_match1_.CalculateIroas(-0.25).status(),
            absl::InvalidArgumentError(
                "Trim rate must be in (0,0.25), but got -0.25"));
}

TEST_F(EstimatorInternalTest, CalculateIroasNoTrim) {
  double total_delta_response = 0.0, total_delta_cost = 0.0;

  for (size_t i = 0; i < delta_response1_.size(); ++i) {
    total_delta_response += delta_response1_[i];
    total_delta_cost += delta_cost1_[i];
  }

  EXPECT_NEAR(total_delta_response / total_delta_cost,
              *trimmed_match1_.CalculateIroas(0.0), 1e-6);
}

TEST_F(EstimatorInternalTest, CalculateIroasWithTrim) {
  double total_delta_response = 0.0, total_delta_cost = 0.0;

  // First two pairs are trimmed.
  for (size_t i = 2; i < delta_response1_.size(); ++i) {
    total_delta_response += delta_response1_[i];
    total_delta_cost += delta_cost1_[i];
  }

  EXPECT_NEAR(total_delta_response / total_delta_cost,
              *trimmed_match1_.CalculateIroas(0.25), 1e-6);
}

TEST_F(EstimatorInternalTest, CalculateStandardErrorNoTrim) {
  const double iroas = 0.0;
  double sum_square_delta_response = 0.0;
  double sum_delta_cost = 0.0;
  for (size_t i = 0; i < delta_response1_.size(); ++i) {
    sum_square_delta_response += Square(delta_response1_[i]);
    sum_delta_cost += delta_cost1_[i];
  }

  EXPECT_NEAR(std::sqrt(sum_square_delta_response) / sum_delta_cost,
              trimmed_match1_.CalculateStandardError(0.0, iroas), 1e-6);
}

TEST_F(EstimatorInternalTest, CalculateStandardErrorWithTrim) {
  const double iroas = 1.0;
  double sum_square_delta_response = 0.0;
  double sum_delta_cost = 0.0;

  // First two pairs are trimmed.
  for (size_t i = 2; i < delta_response1_.size(); ++i) {
    sum_square_delta_response +=
        Square(delta_response1_[i] - iroas * delta_cost1_[i]) * 2;
    sum_delta_cost += delta_cost1_[i];
  }

  EXPECT_NEAR(std::sqrt(sum_square_delta_response) / sum_delta_cost,
              trimmed_match1_.CalculateStandardError(0.25, iroas), 1e-6);
}

TEST_F(EstimatorInternalTest, TrimmedMatch1NoTrim) {
  const Result expected = {
      1.692, 0.138, 0, 0.9, 1.250, 1.905, {{0, 1.692, 0.138}}};
  const auto result = trimmed_match1_.Report(1.64485363, 0.0);

  CompareReports(expected, result, 0.001);
}

TEST_F(EstimatorInternalTest, TrimmedMatch1WithTrim) {
  const Result expected = {
      1.571, 0.041, 0.25, 0.9, -29.375, 1.619, {{0.25, 1.571, 0.041}}};
  const auto result = trimmed_match1_.Report(1.64485363, 0.25);

  CompareReports(expected, result, 0.001);
}

TEST_F(EstimatorInternalTest, TrimmedMatch2NoTrim) {
  const Result expected = {0.005811,
                           0.000117,
                           0,
                           0.9,
                           0.005233,
                           0.006389,
                           {{0, 0.005811, 0.000117}}};
  const auto result = trimmed_match2_.Report(1.64485363, 0.0);

  CompareReports(expected, result);
}

TEST_F(EstimatorInternalTest, TrimmedMatch2WithTrim) {
  const Result expected = {0.005815,
                           0.000134,
                           0.03125,
                           0.9,
                           0.005155,
                           0.006475,
                           {{0.03125, 0.005815, 0.000134}}};
  const auto result = trimmed_match2_.Report(1.64485363, 1.0 / 32);

  CompareReports(expected, result);
}

TEST_F(EstimatorInternalTest, TrimmedMatch2WithOptimTrim) {
  const std::vector<TrimAndError> candidates = {
      {0, 0.00581123, 0.000117108},   {0.03125, 0.005815, 0.000133689},
      {0.0625, 0.00582772, 0.000094}, {0.09375, 0.00583428, 0.000073},
      {0.125, 0.00583461, 0.000062},  {0.15625, 0.00583399, 0.000061},
      {0.1875, 0.00583305, 0.000030}, {0.21875, 0.00583258, 0.000031},
      {0.25, 0.005832, 0.000029},     {0.28125, 0.00585823, 0.000069}};

  const Result expected = {0.005833, 0.00003,  0.1875,    0.9,
                           0.005684, 0.005982, candidates};
  const auto result = trimmed_match2_.Report(1.64485363, -1);

  CompareReports(expected, result);
}

}  // namespace
}  // namespace trimmedmatch
