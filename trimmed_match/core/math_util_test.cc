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
#include "trimmed_match/core/math_util.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

namespace trimmedmatch {
namespace {
class TrimOpsInternalTest : public ::testing::Test {
 protected:
  TrimOpsInternalTest() : residuals_({6.0, 0.0, -3.0}) {}

  const std::vector<double> residuals_;
};

TEST_F(TrimOpsInternalTest, TrimmedSymmetricNormInvalidTrimRate) {
  EXPECT_DEATH(auto result = TrimmedSymmetricNorm(residuals_, -0.1), "");
  EXPECT_DEATH(auto result = TrimmedSymmetricNorm(residuals_, 0.5), "");
}

TEST_F(TrimOpsInternalTest, TrimmedSymmetricNormNoTrim) {
  const double expected = 2.0;
  EXPECT_EQ(expected, TrimmedSymmetricNorm(residuals_, 0.0));
}

TEST_F(TrimOpsInternalTest, TrimmedSymmetricNormTrim) {
  const double expected = 0.0;
  EXPECT_EQ(expected, TrimmedSymmetricNorm(residuals_, 0.1));
}

TEST_F(TrimOpsInternalTest, StudentizedTrimmedMeanInvalidTrimRate) {
  EXPECT_DEATH(auto result = StudentizedTrimmedMean(residuals_, -0.1), "");
  EXPECT_DEATH(auto result = StudentizedTrimmedMean(residuals_, 0.5), "");
}

TEST_F(TrimOpsInternalTest, StudentizedTrimmedMeanNoTrim) {
  const double expected = 0.378;
  EXPECT_NEAR(expected, StudentizedTrimmedMean(residuals_, 0.0), 0.001);
}

TEST_F(TrimOpsInternalTest, StudentizedTrimmedMeanTrim) {
  std::vector<double> res(residuals_);
  res.push_back(10);
  res.push_back(20);
  const double expected = 1.301;

  EXPECT_NEAR(expected, StudentizedTrimmedMean(res, 0.1), 0.001);
}

TEST(QuadraticInequalityTest, GetValueAt) {
  // f(x) = x * x - 2 * x + 1.0
  QuadraticInequality quadratic_inequality(1.0, 1.0, 1.0);
  EXPECT_EQ(1.0, quadratic_inequality.GetValueAt(0.0));
  EXPECT_EQ(0.0, quadratic_inequality.GetValueAt(1.0));
  EXPECT_EQ(1.0, quadratic_inequality.GetValueAt(2.0));
}

TEST(QuadraticInequalityTest, SolverConstantNaN) {
  // f(x) = -1
  QuadraticInequality quadratic_inequality(0.0, 0.0, -1.0);

  const std::pair<double, double> result =
      quadratic_inequality.Solver(0.0, 1.0);
  EXPECT_TRUE(std::isnan(result.first) && std::isnan(result.second));
}

TEST(QuadraticInequalityTest, SolverConstantSame) {
  // f(x) = 1
  QuadraticInequality quadratic_inequality(0.0, 0.0, 1.0);

  const std::pair<double, double> result =
      quadratic_inequality.Solver(0.0, 1.0);
  EXPECT_EQ(std::make_pair(0.0, 1.0), result);
}

TEST(QuadraticInequalityTest, SolverLinear) {
  // f(x) = 2 * x - 3
  QuadraticInequality quadratic_inequality(0.0, -1.0, -3.0);

  const std::pair<double, double> result_nan =
      quadratic_inequality.Solver(0.0, 1.0);
  EXPECT_TRUE(std::isnan(result_nan.first) && std::isnan(result_nan.second));

  const std::pair<double, double> result_point =
      quadratic_inequality.Solver(-1.0, 1.5);
  EXPECT_EQ(std::make_pair(1.5, 1.5), result_point);

  const std::pair<double, double> result_same =
      quadratic_inequality.Solver(2.0, 3.0);
  EXPECT_EQ(std::make_pair(2.0, 3.0), result_same);

  const std::pair<double, double> result_interval =
      quadratic_inequality.Solver(-2.0, 2.0);
  EXPECT_EQ(std::make_pair(1.5, 2.0), result_interval);
}

TEST(QuadraticInequalityTest, SolverQuadratic) {
  // f(x) = x * x - 2 * x - 3
  QuadraticInequality quadratic_inequality(1.0, 1.0, -3.0);

  const std::pair<double, double> result_nan =
      quadratic_inequality.Solver(0.0, 1.0);
  EXPECT_TRUE(std::isnan(result_nan.first) && std::isnan(result_nan.second));

  const std::pair<double, double> result_point =
      quadratic_inequality.Solver(-1.0, 2.0);
  EXPECT_EQ(std::make_pair(-1.0, -1.0), result_point);

  const std::pair<double, double> result_same =
      quadratic_inequality.Solver(10.0, 20.0);
  EXPECT_EQ(std::make_pair(10.0, 20.0), result_same);

  const std::pair<double, double> result_interval =
      quadratic_inequality.Solver(-2.0, 2.0);
  EXPECT_EQ(std::make_pair(-2.0, -1.0), result_interval);
}

}  // namespace
}  // namespace trimmedmatch
