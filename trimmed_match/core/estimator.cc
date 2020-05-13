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
// TrimmedMatch estimator.

#include "trimmed_match/core/estimator.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "glog/logging.h"
#include "absl/types/optional.h"
#include "boost/math/distributions/students_t.hpp"
#include "trimmed_match/core/geox_data_util.h"
#include "trimmed_match/core/math_util.h"

namespace trimmedmatch {

namespace {

// Maximum trim rate.
static double kMaxTrimRate = 0.30;

// If ci width is > std.error * this value, uses 3 times normal quantile.
// It is possible that the confidence interval calculated from the inequality
// (5.7) in https://ai.google/research/pubs/pub48448/ may be (-infty, infty),
// for which three times of the Normal-approximate CI width is used.
static double kExtremeCiWidth = 1e6;
static double kNormalMultiplier = 3.0;

struct ResidualAndDeltaCost {
  double res;
  double delta_cost;
};

absl::optional<GeoxDataUtil> InitializeUtilFromRawData(
    const std::vector<double>& delta_response,
    const std::vector<double>& delta_cost) {
  if (delta_response.size() != delta_cost.size()) {
    return absl::nullopt;
  }
  std::vector<GeoPairValues> geox_data(delta_response.size());
  for (size_t i = 0; i < delta_response.size(); ++i) {
    geox_data[i] = {delta_response[i], delta_cost[i]};
  }

  return GeoxDataUtil(geox_data);
}

}  // namespace

TrimmedMatch::TrimmedMatch(const std::vector<double>& delta_response,
                           const std::vector<double>& delta_cost,
                           const double max_trim_rate)
    : max_trim_rate_(max_trim_rate >= kMaxTrimRate
                         ? kMaxTrimRate
                         : std::max(max_trim_rate, 0.0)),
      num_pairs_(static_cast<int>(delta_response.size())),
      geox_util_(InitializeUtilFromRawData(delta_response, delta_cost)) {
  CHECK(geox_util_ != absl::nullopt)
      << "delta_response and delta_cost must have the same length, but got "
      << delta_response.size() << " vs " << delta_cost.size();
}

double TrimmedMatch::CalculateIroas(const double trim_rate) const {
  CHECK(trim_rate >= 0.0 && trim_rate <= max_trim_rate_)
      << "Trim rate must be in (0, " << max_trim_rate_ << "), but got "
      << trim_rate;

  if (trim_rate == 0.0) {
    return geox_util_->CalculateEmpiricalIroas();
  }

  const std::vector<double> candidates =
      geox_util_->FindAllZerosOfTrimmedMean(trim_rate);

  CHECK(!candidates.empty())
      << "Incremental cost for the untrimmed geo pairs is 0";

  if (candidates.size() == 1) {
    return candidates[0];
  }

  // If multiple candidates exist, finds one that minimizes TrimmedSymNorm.
  double trimmed_symmetric_norm = std::numeric_limits<double>::max();
  double result = 0.0;
  for (size_t i = 0; i < candidates.size(); ++i) {
    const std::vector<double> residuals =
        geox_util_->CalculateResiduals(candidates[i]);
    const double candidate_norm = TrimmedSymmetricNorm(residuals, trim_rate);
    if (candidate_norm < trimmed_symmetric_norm) {
      trimmed_symmetric_norm = candidate_norm;
      result = candidates[i];
    }
  }

  return result;
}

double TrimmedMatch::CalculateStandardError(const double trim_rate,
                                            const double iroas) const {
  CHECK(trim_rate >= 0.0 && trim_rate <= max_trim_rate_)
      << "Trim rate must be in (0, " << max_trim_rate_ << "), but got "
      << trim_rate;
  const std::vector<double> res = geox_util_->CalculateResiduals(iroas);
  const int n1 = static_cast<int>(std::ceil(trim_rate * num_pairs_));
  const int n2 = num_pairs_ - 1 - n1;

  std::vector<ResidualAndDeltaCost> res_delta_cost(num_pairs_);
  const auto delta_cost = geox_util_->ExtractDeltaCost();
  for (int i = 0; i < num_pairs_; ++i) {
    res_delta_cost[i] = {res[i], delta_cost[i]};
  }
  std::sort(res_delta_cost.begin(), res_delta_cost.end(),
            [](const ResidualAndDeltaCost& a, const ResidualAndDeltaCost& b) {
              return (a.res < b.res);
            });

  double tr_mean_delta_cost = 0.0;

  // Calculates winsorized squared sum of res and trimmed squared sum of
  // delta_cost. See Eq (8.2) in https://ai.google/research/pubs/pub48448/.
  double squared_sum_res =
      n1 * (Square(res_delta_cost[n1].res) + Square(res_delta_cost[n2].res));
  for (int i = n1; i <= n2; ++i) {
    tr_mean_delta_cost += res_delta_cost[i].delta_cost;
    squared_sum_res += Square(res_delta_cost[i].res);
  }
  tr_mean_delta_cost /= num_pairs_;

  double approx_variance =
      squared_sum_res / (Square(tr_mean_delta_cost) * num_pairs_);

  return std::sqrt(approx_variance / num_pairs_);
}

Result TrimmedMatch::Report(const double confidence,
                            const double trim_rate) const {
  TrimAndError result;
  std::vector<TrimAndError> candidate_results;

  // If trim_rate falls into [0, max_trim_rate_), use it.
  // Otherwise, choose a trim rate in that range so that the corresponding
  // standard error of the estimate is close to the minimum.
  if (trim_rate >= 0.0 && trim_rate <= max_trim_rate_) {
    const double iroas = CalculateIroas(trim_rate);
    const double std_error = CalculateStandardError(trim_rate, iroas);
    result = {trim_rate, iroas, std_error};
    candidate_results.push_back(result);
  } else {
    const int max_num_trim =
        static_cast<int>(std::ceil(max_trim_rate_ * num_pairs_));
    for (int i = 0; i <= max_num_trim; ++i) {
      const double rate = static_cast<double>(i) / num_pairs_;
      if (rate > max_trim_rate_) break;
      const double iroas = CalculateIroas(rate);
      const double std_error = CalculateStandardError(rate, iroas);
      candidate_results.push_back({rate, iroas, std_error});
    }

    // Choose the result close to, but no more than 1 + 0.25/sqrt(num_pairs)
    // times of the minimum std.error.
    const double min_error =
        std::min_element(candidate_results.begin(), candidate_results.end(),
                         [&](const TrimAndError& a, const TrimAndError& b) {
                           return (a.std_error < b.std_error);
                         })
            ->std_error;
    for (const auto& candidate : candidate_results) {
      if (candidate.std_error <=
          (1.0 + 0.25 / std::sqrt(num_pairs_)) * min_error) {
        result = candidate;
        break;
      }
    }
  }

  // Confidence interval.
  boost::math::normal dist(0.0, 1.0);
  const double q = boost::math::quantile(dist, 0.5 + 0.5 * confidence);
  std::pair<double, double> ci =
      geox_util_->RangeFromStudentizedTrimmedMean(result.trim_rate, q);

  if (ci.second - ci.first > kExtremeCiWidth * result.std_error) {
    ci.first = result.iroas - result.std_error * q * kNormalMultiplier;
    ci.second = result.iroas + result.std_error * q * kNormalMultiplier;
  }

  return {result.iroas, result.std_error, result.trim_rate, confidence,
          ci.first,     ci.second,        candidate_results};
}

}  // namespace trimmedmatch
