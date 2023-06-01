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
// Trimmed Match estimator of Incremental Return on Ad Spend (iROAS) for GeoX
// post analysis with a randomized paired design.
// Reports the point estimate, std.error, optimal trim rate and confidence
// intervals. See the tech details at https://ai.google/research/pubs/pub48448/.

#ifndef TRIMMED_MATCH_CORE_ESTIMATOR_H_
#define TRIMMED_MATCH_CORE_ESTIMATOR_H_

#include <limits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/types/optional.h"
#include "trimmed_match/core/geox_data_util.h"
#include "trimmed_match/core/math_util.h"

namespace trimmedmatch {

struct TrimAndError {
  double trim_rate;  // Trim rate
  double iroas;      // Incremental return on ad spend, point estimate
  double std_error;  // Standard error
};

struct Result {
  double estimate;           // Point estimate
  double std_error;          // Standard error of the point estimate
  double trim_rate;          // Trim rate learned from data
  double normal_quantile;    // Normal quantile at (.5 + .5 * confidence), where
                             // confidence is the level of 2-sided conf interval
  double conf_interval_low;  // Confidence interval lower bound
  double conf_interval_up;   // Confidence interval upper bound

  // Results for candidate trim rates
  std::vector<TrimAndError> candidate_results;
};

class TrimmedMatch {
 public:
  // This class produces the iROAS estimate, standard error and confidence
  // interval, as well as optimal trim rate (if not provided), with input
  // data from a randomized paired geo experiment.
  // delta_response is the difference of responses between treated geo and
  // controlled geo for each pair; similarly delta_cost is defined.
  // Example use:
  //   TrimmedMatch tm(delta_response, delta_cost);
  //   Report report = tm.Report();
  //   std::cout << "iROAS point estimate=" << report.estimate
  //             << "with CI [" << report.conf_interval_low
  //             << ", " << report.conf_interval_up << "]" << std::endl;
  explicit TrimmedMatch(const std::vector<double>& delta_response,
                        const std::vector<double>& delta_cost,
                        double max_trim_rate = 0.25);

  // Returns the root of the trimmed mean equation which minimizes
  // TrimmedSymmetricNorm().
  absl::StatusOr<double> CalculateIroas(double trim_rate) const;

  // Returns the square root of (asymptotic variance / number of pairs), where
  // asymptotic variance is given by Eq (8.2) in
  // https://ai.google/research/pubs/pub48448/.
  double CalculateStandardError(double trim_rate, double iroas) const;

  // Returns Report of the iROAS estimation.
  // If trim_rate is not provided or is not in [0, max_trim_rate], calculate the
  // result for each candidate trim rate and report the result for the smallest
  // trim rate such that its corresponding std.error is less than
  // (1 + 0.25 / sqrt(num of pairs)) * minimum std.error.
  // Otherwise, report the result for the given trim rate.
  // Normal_quantile is by default the 90% normal percentile, which corresponds
  // to 80% 2-sided confidence interval.
  absl::StatusOr<Result> Report(double normal_quantile = 1.281551566,
                                double trim_rate = -1.0) const;

 private:
  const double max_trim_rate_;
  const int num_pairs_;
  absl::optional<GeoxDataUtil> geox_util_;
};

}  // namespace trimmedmatch

#endif  // TRIMMED_MATCH_CORE_ESTIMATOR_H_
