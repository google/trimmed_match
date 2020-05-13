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
// A class to organize data from paired randomized geo experiments, and
// compute a point estimate and confidence interval of incremental ROAS
// from trimmed match as described at https://ai.google/research/pubs/pub48448/.

#ifndef TRIMMED_MATCH_CORE_GEOX_DATA_UTIL_H_
#define TRIMMED_MATCH_CORE_GEOX_DATA_UTIL_H_

#include <limits>
#include <vector>

#include "trimmed_match/core/math_util.h"

namespace trimmedmatch {

// Variation less than 1e-10 USD in delta_cost can be ignored.
static double kTieBreaker = 1e-10;

// (difference of response, difference of costs) within a pair of geos, one
// for treatment and the other for control, where the difference is defined as
// treatment - control.
struct GeoPairValues {
  double delta_response;
  double delta_cost;
};

// Identifies (i, j, delta), where
// delta = (delta_response[i] - delta_response[j]) / (delta_cost[i] -
// delta_cost[j]).
struct PairedDelta {
  int i;
  int j;
  double delta;
};

// Some utility functions to organize geox data and solve the trimmed mean
// equation with a user-specified trim rate.
class GeoxDataUtil {
 public:
  explicit GeoxDataUtil(const std::vector<GeoPairValues>& geox_data);

  // Calculates the ratio of two numbers.
  static double CalculateRatio(double numerator, double denominator) {
    return (numerator / (denominator == 0.0 ? std::numeric_limits<double>::min()
                                            : denominator));
  }

  // Extracts geox data.
  std::vector<GeoPairValues> ExtractGeoxData() const;

  // Extracts the delta_cost vector from GeoPairValues.
  std::vector<double> ExtractDeltaCost() const;

  // Extracts the delta_response vector from GeoPairValues.
  std::vector<double> ExtractDeltaResponse() const;

  // Extracts paired_delta_sorted.
  std::vector<PairedDelta> ExtractPairedDelta() const;

  // Calculates residuals = delta_response - delta_cost * iroas for each geo
  // pair.
  std::vector<double> CalculateResiduals(double iroas) const;

  // Solves the trimmed mean equation and finds all zeros.
  //     TrimmedMean(delta_response - delta_cost * delta) = 0.
  std::vector<double> FindAllZerosOfTrimmedMean(double trim_rate) const;

  // Solves the studentized trimmed mean inequality w.r.t. delta:
  //   |StudentizedTrimmedMean(delta_response - delta * delta_cost)|
  //      <= threshold
  // which is defined by Eq.(5.6) in https://ai.google/research/pubs/pub48448/.
  std::pair<double, double> RangeFromStudentizedTrimmedMean(
      double trim_rate, double threshold) const;

  // Calculates the empirical estimate of IROAS as the ratio of
  // sum(delta_response) and sum(delta_cost) across geo pairs.
  double CalculateEmpiricalIroas() const;

 private:
  struct DeltaMinMax {
    double delta_min;
    double delta_max;
  };

  // Compares k-th geo pair with each other geo pair and calculates the ratio of
  // delta_response difference and delta_cost difference, then outputs the min
  // and max of the ratio values.
  DeltaMinMax DeltaRelativeToOneGeoPair(int index) const;

  // Evaluates the range of iROAS.
  DeltaMinMax DeltaRange(int n1, int n2) const;

  // Sorted geox data.
  const int num_pairs_;
  std::vector<GeoPairValues> geox_data_;
  std::vector<PairedDelta> paired_delta_sorted_;
};

// Identifies the first jump in delta_cost from a given location.
int FindFirstJumpInDeltaCostFrom(const std::vector<GeoPairValues>& geox_data,
                                 int i);

// Creates candidate deltas in the format of PairedDelta.
std::vector<PairedDelta> GetPairedDeltaSorted(
    const std::vector<GeoPairValues>& geox_data);

}  // namespace trimmedmatch

#endif  // TRIMMED_MATCH_CORE_GEOX_DATA_UTIL_H_
