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

#include "glog/logging.h"

namespace trimmedmatch {

double TrimmedSymmetricNorm(const std::vector<double>& residuals,
                            const double trim_rate) {
  CHECK(trim_rate >= 0.0 && trim_rate < 0.5)
      << "Trim_rate must be in (0, 0.5), but got " << trim_rate;

  size_t size = residuals.size();
  size_t n_trim_left = static_cast<size_t>(std::ceil(size * trim_rate));
  CHECK(2 * n_trim_left < size)
      << "After trimming, no data is left -- got trim_rate " << trim_rate;

  std::vector<double> residuals_copy(residuals);
  std::nth_element(residuals_copy.begin(), residuals_copy.begin() + n_trim_left,
                   residuals_copy.end());
  std::partial_sort(residuals_copy.begin() + n_trim_left,
                    residuals_copy.end() - n_trim_left,
                    residuals_copy.end());

  double sum_norm = 0.0;
  for (size_t i = n_trim_left; i < size - n_trim_left; ++i) {
    sum_norm += std::abs(residuals_copy[i] + residuals_copy[size - 1 - i]);
  }

  return sum_norm / (size - 2 * n_trim_left);
}

double StudentizedTrimmedMean(const std::vector<double>& residuals,
                              const double trim_rate) {
  CHECK(trim_rate >= 0.0 && trim_rate < 0.5)
      << "trim_rate must be in [0, 0.5), but got " << trim_rate;
  CHECK(!residuals.empty()) << "residuals is empty";
  const int size = static_cast<int>(residuals.size());
  const int n_left = static_cast<int>(std::ceil(trim_rate * size));
  CHECK(2 * n_left + 1 < size)
      << "At least 2 values must be left after trimming, but got "
      << size - 2 * n_left;

  std::vector<double> res_copy(residuals);
  std::nth_element(res_copy.begin(), res_copy.begin() + n_left, res_copy.end());
  std::nth_element(res_copy.begin() + n_left, res_copy.end() - n_left,
                   res_copy.end());

  double trim_sum = 0.0;
  for (int i = n_left; i < size - n_left; ++i) {
    trim_sum += res_copy[i];
  }

  double winsorized_mean =
      (trim_sum + n_left * (res_copy[n_left] + res_copy[size - n_left - 1])) /
      size;
  double sum_winsorized_squares =
      n_left * (Square(res_copy[n_left] - winsorized_mean) +
                Square(res_copy[size - n_left - 1] - winsorized_mean));
  for (int i = n_left; i < size - n_left; ++i) {
    sum_winsorized_squares += Square(res_copy[i] - winsorized_mean);
  }

  return (trim_sum / (size - 2 * n_left)) /
         std::sqrt(sum_winsorized_squares / (size - 2 * n_left) /
                   (size - 2 * n_left - 1));
}

QuadraticInequality::QuadraticInequality(const double a, const double b,
                                         const double c, const double bound)
    : a_(a), b_(b), c_(c), bound_(bound) {}

double QuadraticInequality::GetValueAt(const double x) const {
  double x_value = std::min(bound_, std::max(-bound_, x));
  return a_ * x_value * x_value - 2 * b_ * x_value + c_;
}

std::pair<double, double> QuadraticInequality::Solver(
    const double min_x, const double max_x) const {
  CHECK(min_x <= max_x) << "min_x cannot be greater than max_x, but got min_x="
                        << min_x << " and max_x=" << max_x;

  const double min_value = std::max(-bound_, min_x);
  const double max_value = std::min(bound_, max_x);
  const double f1 = GetValueAt(min_value);
  const double f2 = GetValueAt(max_value);

  if (f1 >= 0.0 && f2 >= 0.0) {
    return {min_value, max_value};
  }

  // If f1 and f2 have different signs, then a_ and b_ cannot be both 0.0, and
  // the determinant must be nonnegative.
  if (f1 >= 0.0 && f2 < 0.0) {
    const double x_right = ((a_ == 0.0) ? (0.5 * c_ / b_)
                            : ((b_ - std::sqrt(b_ * b_ - a_ * c_)) / a_));
    return {min_value, x_right};
  }

  if (f1 < 0.0 && f2 >= 0.0) {
    const double x_left = ((a_ == 0.0) ? (0.5 * c_ / b_)
                           : ((b_ + std::sqrt(b_ * b_ - a_ * c_)) / a_));
    return {x_left, max_value};
  }

  const double discriminant = b_ * b_ - a_ * c_;
  if (discriminant < 0.0 || a_ >= 0.0) {
    return {kNaN, kNaN};
  }

  const double center = b_ / a_;
  if (center >= min_value && center <= max_value) {
    const double shift = std::sqrt(discriminant) / a_;
    return {center + shift, center - shift};
  }

  return {kNaN, kNaN};
}

}  // namespace trimmedmatch
