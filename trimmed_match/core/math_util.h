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
// Some math utility functions.

#ifndef TRIMMED_MATCH_CORE_MATH_UTIL_H_
#define TRIMMED_MATCH_CORE_MATH_UTIL_H_

#include <limits>
#include <vector>

namespace trimmedmatch {

static double kNaN = std::numeric_limits<double>::quiet_NaN();

// Bound for iROAS.
static double kBoundIroas = 1e10;

// The Square function.
inline double Square(const double x) { return x * x; }

// Measures the symmetric deviation from 0.
// Given x1, x2, ..., xn, sorts them such that x(1) <= x(2) <= ... <= x(n),
// then trims tails from both ends and calculates the average of
// |x(i) + x(n-i)| for i not trimmed.
double TrimmedSymmetricNorm(const std::vector<double>& residuals,
                            double trim_rate);

// Calculates the studentized trimmed mean.
// StudentizedTrimmedMean is trimmed mean divided by Square root of
// its variance estimate, see the formula in for example Section 3 of
// https://www.mat.ulaval.ca/fileadmin/mat/documents/lrivest/Publications/34-CaperaaRivest1995.pdf)
double StudentizedTrimmedMean(const std::vector<double>& residuals,
                              double trim_rate);

// A class for solving the quadratic inequality with inequality constraints:
//   a * x * x - 2 * b * x + c >= 0, where min_x <= x <= max_x and |x| <= bound.
class QuadraticInequality {
 public:
  explicit QuadraticInequality(double a, double b, double c,
                               double bound = kBoundIroas);

  // Calculates a * x * x - 2 * b * x + c.
  double GetValueAt(double x) const;

  // Returns the minimum and maximum x values that satisfy the inequalities, and
  // {kNaN, kNaN} if no solution exists.
  std::pair<double, double> Solver(double min_x, double max_x) const;

 private:
  const double a_;
  const double b_;
  const double c_;
  const double bound_;
};

}  // namespace trimmedmatch

#endif  // TRIMMED_MATCH_CORE_MATH_UTIL_H_
