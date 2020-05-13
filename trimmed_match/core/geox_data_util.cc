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

#include "glog/logging.h"
#include "trimmed_match/core/math_util.h"

namespace trimmedmatch {

GeoxDataUtil::GeoxDataUtil(const std::vector<GeoPairValues>& geox_data)
    : num_pairs_(static_cast<int>(geox_data.size())) {
  // Sort geox data by delta_cost.
  for (int i = 0; i < num_pairs_; ++i) {
    geox_data_.push_back(geox_data[i]);
  }
  std::sort(geox_data_.begin(), geox_data_.end(),
            [](const GeoPairValues& a, const GeoPairValues& b) {
              return (a.delta_cost < b.delta_cost);
            });

  // Break ties if any.
  int i = 1;
  while (i < num_pairs_) {
    if (geox_data_[i].delta_cost - geox_data_[i - 1].delta_cost <=
        std::numeric_limits<double>::min()) {
      const int j = FindFirstJumpInDeltaCostFrom(geox_data_, i);
      for (int k = i; k < j; ++k) {
        geox_data_[k].delta_cost = geox_data_[i - 1].delta_cost +
                                   (kTieBreaker * (k - i + 1)) / (j - i + 1);
      }
      i = j + 1;
    } else {
      i++;
    }
  }

  paired_delta_sorted_ = GetPairedDeltaSorted(geox_data_);
}

std::vector<GeoPairValues> GeoxDataUtil::ExtractGeoxData() const {
  return geox_data_;
}

std::vector<double> GeoxDataUtil::ExtractDeltaCost() const {
  std::vector<double> delta_cost(num_pairs_);
  for (int i = 0; i < num_pairs_; ++i) {
    delta_cost[i] = geox_data_[i].delta_cost;
  }
  return delta_cost;
}

std::vector<double> GeoxDataUtil::ExtractDeltaResponse() const {
  std::vector<double> delta_response(num_pairs_);
  for (int i = 0; i < num_pairs_; ++i) {
    delta_response[i] = geox_data_[i].delta_response;
  }
  return delta_response;
}

std::vector<PairedDelta> GeoxDataUtil::ExtractPairedDelta() const {
  return paired_delta_sorted_;
}

std::vector<double> GeoxDataUtil::FindAllZerosOfTrimmedMean(
    const double trim_rate) const {
  CHECK(num_pairs_ >= 1) << "No data";
  CHECK(trim_rate >= 0.0 && trim_rate < 0.5)
      << "Trim rate must be in [0.0, 0.5), but got " << trim_rate;

  std::vector<double> estimates;
  if (trim_rate == 0.0) {
    estimates.push_back(CalculateEmpiricalIroas());
    return estimates;
  }

  // Solves the equation for delta such that
  //   TrimmedMean({delta_response[i] - delta * delta_cost[i]: i=1...n}) = 0.
  // Assumes that delta_costs are sorted.
  // Let e[i] = delta_response[i] - delta * delta_cost[i], i=1, ..., n.
  // When delta is too close to -infinity or infinity, the order of e[i]
  // is the same as delta_cost[i] (or -delta_cost[i]). So we know which to trim.
  // When delta is in the middle, we identify all small intervals such that
  // within each interval, the order of e[i]'s do not change and thus there is
  // at most 1 zero. This leads to the interval boundaries defined by
  // (delta_response[j] - delta_response[i]) / (delta_cost[j] - delta_cost[i]).
  // Sorts the n-choose-2 boundaries (associated with the indices) and
  // checks them from left to right.

  // Estimates based on the range [n1, n2].
  const int n1 = static_cast<int>(std::ceil(trim_rate * num_pairs_));
  CHECK(2 * n1 < num_pairs_)
      << "Num of trimmed values must be strictly less than " << num_pairs_
      << ", but got " << 2 * n1;

  const int n2 = num_pairs_ - 1 - n1;
  std::vector<bool> body_set(num_pairs_, false);

  // Identifies the range of the estimate.
  const auto delta_range = DeltaRange(n1, n2);

  // Initializes the body set and tracks the sum of delta_responses and
  // delta_costs.
  double sum_delta_response = 0.0;
  double sum_delta_cost = 0.0;
  for (int i = n1; i <= n2; ++i) {
    body_set[i] = true;
    sum_delta_response += geox_data_[i].delta_response;
    sum_delta_cost += geox_data_[i].delta_cost;
  }

  // Scans the range from paired_delta_sorted_ sequentially.
  std::vector<double> candidates;
  std::vector<double> bounds;

  const double value = CalculateRatio(sum_delta_response, sum_delta_cost);
  candidates.push_back(value);
  bounds.push_back(-std::numeric_limits<double>::max());

  for (const auto& x : paired_delta_sorted_) {
    if ((x.delta < delta_range.delta_min) ||
        (x.delta > delta_range.delta_max)) {
      continue;
    }
    if (body_set[x.i] == body_set[x.j]) continue;

    // Updates iROAS and body_set.
    const double temp_dresp =
        geox_data_[x.j].delta_response - geox_data_[x.i].delta_response;
    const double temp_delta_cost =
        geox_data_[x.j].delta_cost - geox_data_[x.i].delta_cost;
    if (body_set[x.i]) {
      sum_delta_response += temp_dresp;
      sum_delta_cost += temp_delta_cost;
    } else {
      sum_delta_response -= temp_dresp;
      sum_delta_cost -= temp_delta_cost;
    }
    body_set[x.i] = !body_set[x.i];
    body_set[x.j] = !body_set[x.j];

    candidates.push_back(CalculateRatio(sum_delta_response, sum_delta_cost));
    bounds.push_back(x.delta);
  }

  // Identifies valid estimates.
  bounds.push_back(std::numeric_limits<double>::max());
  for (int i = 0; i < static_cast<int>(candidates.size()); ++i) {
    if ((candidates[i] >= bounds[i]) && (candidates[i] <= bounds[i + 1])) {
      estimates.push_back(candidates[i]);
    }
  }

  return estimates;
}

std::vector<double> GeoxDataUtil::CalculateResiduals(const double iroas) const {
  std::vector<double> residuals(num_pairs_);
  for (int i = 0; i < num_pairs_; ++i) {
    residuals[i] =
        geox_data_[i].delta_response - geox_data_[i].delta_cost * iroas;
  }
  return residuals;
}

std::pair<double, double> GeoxDataUtil::RangeFromStudentizedTrimmedMean(
    const double trim_rate, const double threshold) const {
  CHECK(trim_rate >= 0.0 && threshold > 0.0 && num_pairs_ >= 2)
      << "\nTrim rate must be in [0, 0.5):" << trim_rate
      << "\nthreshold must be positive:" << threshold
      << "num_pairs must be greater than 2:" << num_pairs_;

  if (trim_rate == 0.0) {
    const double H = (threshold * threshold * num_pairs_) / (num_pairs_ - 1);
    double X1 = 0.0, Y1 = 0.0, X2 = 0.0, Y2 = 0.0, Z = 0.0;
    for (const auto& x : geox_data_) {
      X1 += x.delta_cost;
      Y1 += x.delta_response;
      X2 += Square(x.delta_cost);
      Y2 += Square(x.delta_response);
      Z += x.delta_response * x.delta_cost;
    }
    const double r = 1.0 + H / num_pairs_;
    QuadraticInequality qi(H * X2 - r * X1 * X1, H * Z - r * X1 * Y1,
                           H * Y2 - r * Y1 * Y1);
    return qi.Solver(-std::numeric_limits<double>::max(),
                     std::numeric_limits<double>::max());
  }

  const int n1 = static_cast<int>(std::ceil(trim_rate * num_pairs_));
  CHECK(num_pairs_ >= 2 * n1 + 2)
      << "Less than 2 values are left after trimming";

  // First finds all candidates for each small interval,
  // then checks whether they fall into the interval,
  // and returns the range for the valid ones.
  const int n2 = num_pairs_ - 1 - n1;
  const auto delta_range = DeltaRange(n1, n2);
  const double H = (threshold * threshold * (num_pairs_ - 2 * n1)) /
                   (num_pairs_ - 2 * n1 - 1);
  int left = n1;
  int right = n2;
  double X1 = 0.0, Y1 = 0.0;
  double X2 = n1 * (Square(geox_data_[left].delta_cost) +
                    Square(geox_data_[right].delta_cost));
  double Y2 = n1 * (Square(geox_data_[left].delta_response) +
                    Square(geox_data_[right].delta_response));
  double Z =
      n1 * (geox_data_[left].delta_response * geox_data_[left].delta_cost +
            geox_data_[right].delta_response * geox_data_[right].delta_cost);

  std::vector<bool> body_set(num_pairs_, false);
  for (int i = n1; i <= n2; ++i) {
    body_set[i] = true;
    X1 += geox_data_[i].delta_cost;
    Y1 += geox_data_[i].delta_response;
    X2 += Square(geox_data_[i].delta_cost);
    Y2 += Square(geox_data_[i].delta_response);
    Z += geox_data_[i].delta_response * geox_data_[i].delta_cost;
  }

  double X1w =
      X1 + n1 * (geox_data_[left].delta_cost + geox_data_[right].delta_cost);
  double Y1w = Y1 + n1 * (geox_data_[left].delta_response +
                          geox_data_[right].delta_response);

  // Scans the range from paired_delta_sorted_ sequentially.
  std::vector<std::pair<double, double>> candidates;
  QuadraticInequality qi(H * (X2 - X1w * X1w / num_pairs_) - X1 * X1,
                         H * (Z - X1w * Y1w / num_pairs_) - X1 * Y1,
                         H * (Y2 - Y1w * Y1w / num_pairs_) - Y1 * Y1);
  const auto left_candidate =
      qi.Solver(-std::numeric_limits<double>::max(), delta_range.delta_min);
  const auto right_candidate =
      qi.Solver(delta_range.delta_max, std::numeric_limits<double>::max());

  if (!std::isnan(left_candidate.first)) {
    candidates.push_back(left_candidate);
  }
  if (!std::isnan(right_candidate.first)) {
    candidates.push_back(right_candidate);
  }

  double min_x = delta_range.delta_min;
  for (size_t k = 0, sz = paired_delta_sorted_.size(); k < sz; ++k) {
    const auto item = paired_delta_sorted_[k];
    if (item.delta <
        delta_range.delta_min - std::numeric_limits<double>::min()) {
      continue;
    }
    if (item.delta >
        delta_range.delta_max + std::numeric_limits<double>::min()) {
      break;
    }

    // Searches for candidate if anything update from left, right or body_set.
    if ((item.i == left) || (item.j == left) || (item.i == right) ||
        (item.j == right) || !(body_set[item.i] == body_set[item.j])) {
      QuadraticInequality candidate_qi(
          H * (X2 - X1w * X1w / num_pairs_) - X1 * X1,
          H * (Z - X1w * Y1w / num_pairs_) - X1 * Y1,
          H * (Y2 - Y1w * Y1w / num_pairs_) - Y1 * Y1);
      const auto candidate = candidate_qi.Solver(min_x, item.delta);
      if (!std::isnan(candidate.first)) {
        candidates.push_back(candidate);
      }
      min_x = item.delta;
    }

    // Updates left and right.
    for (int loc : {left, right}) {
      if ((item.i == loc) || (item.j == loc)) {
        X2 -= n1 * Square(geox_data_[loc].delta_cost);
        Y2 -= n1 * Square(geox_data_[loc].delta_response);
        Z -= n1 * geox_data_[loc].delta_response * geox_data_[loc].delta_cost;
        loc = (item.i == loc) ? item.j : item.i;
        X2 += n1 * Square(geox_data_[loc].delta_cost);
        Y2 += n1 * Square(geox_data_[loc].delta_response);
        Z += n1 * geox_data_[loc].delta_response * geox_data_[loc].delta_cost;
      }
    }
    if ((item.i == left) || (item.j == left)) {
      left = (item.i == left) ? item.j : item.i;
    }
    if ((item.i == right) || (item.j == right)) {
      right = (item.i == right) ? item.j : item.i;
    }

    X1w =
        X1 + n1 * (geox_data_[left].delta_cost + geox_data_[right].delta_cost);
    Y1w = Y1 + n1 * (geox_data_[left].delta_response +
                     geox_data_[right].delta_response);

    // No changes on body_set.
    if (body_set[item.i] == body_set[item.j]) continue;

    // Updates body_set.
    const double temp_delta_response =
        geox_data_[item.j].delta_response - geox_data_[item.i].delta_response;
    const double temp_delta_cost =
        geox_data_[item.j].delta_cost - geox_data_[item.i].delta_cost;
    const double temp_Y2 = Square(geox_data_[item.j].delta_response) -
                           Square(geox_data_[item.i].delta_response);
    const double temp_X2 = Square(geox_data_[item.j].delta_cost) -
                           Square(geox_data_[item.i].delta_cost);
    const double temp_Z =
        (geox_data_[item.j].delta_response * geox_data_[item.j].delta_cost -
         geox_data_[item.i].delta_response * geox_data_[item.i].delta_cost);

    auto UpdateSign = [](bool sign, const double x) { return sign ? x : -x; };

    X1 += UpdateSign(body_set[item.i], temp_delta_cost);
    Y1 += UpdateSign(body_set[item.i], temp_delta_response);
    X2 += UpdateSign(body_set[item.i], temp_X2);
    Y2 += UpdateSign(body_set[item.i], temp_Y2);
    Z += UpdateSign(body_set[item.i], temp_Z);
    X1w += UpdateSign(body_set[item.i], temp_delta_cost);
    Y1w += UpdateSign(body_set[item.i], temp_delta_response);

    body_set[item.i] = !body_set[item.i];
    body_set[item.j] = !body_set[item.j];
  }

  // Identifies the range.
  double ci_low = std::numeric_limits<double>::max();
  double ci_up = -std::numeric_limits<double>::max();
  for (const auto& x : candidates) {
    if (std::isnan(x.first)) continue;
    ci_low = std::min(ci_low, x.first);
    ci_up = std::max(ci_up, x.second);
  }

  if (ci_low > ci_up) {
    return {kNaN, kNaN};
  }

  return {ci_low, ci_up};
}

double GeoxDataUtil::CalculateEmpiricalIroas() const {
  double sum_delta_response = 0.0;
  double sum_delta_cost = 0.0;

  for (int i = 0; i < num_pairs_; ++i) {
    sum_delta_response += geox_data_[i].delta_response;
    sum_delta_cost += geox_data_[i].delta_cost;
  }

  return CalculateRatio(sum_delta_response, sum_delta_cost);
}

GeoxDataUtil::DeltaMinMax GeoxDataUtil::DeltaRelativeToOneGeoPair(
    const int index) const {
  CHECK(index >= 0 && index < num_pairs_)
      << "Index must be in [0, " << num_pairs_ << "), but got " << index;

  double delta_min = std::numeric_limits<double>::max();
  double delta_max = -delta_min;
  for (int i = 0; i < num_pairs_; ++i) {
    if (i != index) {
      double delta = CalculateRatio(
          geox_data_[i].delta_response - geox_data_[index].delta_response,
          geox_data_[i].delta_cost - geox_data_[index].delta_cost);
      if (delta < delta_min) delta_min = delta;
      if (delta > delta_max) delta_max = delta;
    }
  }

  return {delta_min, delta_max};
}

GeoxDataUtil::DeltaMinMax GeoxDataUtil::DeltaRange(const int n1,
                                                   const int n2) const {
  const DeltaMinMax& delta_n1 = DeltaRelativeToOneGeoPair(n1);
  const DeltaMinMax& delta_n2 = DeltaRelativeToOneGeoPair(n2);

  return {std::min(delta_n1.delta_min, delta_n2.delta_min),
          std::max(delta_n1.delta_max, delta_n2.delta_max)};
}

int FindFirstJumpInDeltaCostFrom(const std::vector<GeoPairValues>& geox_data,
                                 const int i) {
  const int num_pairs = static_cast<int>(geox_data.size());

  if (i < 1) return 1;
  if (i >= num_pairs) return num_pairs;

  int j = i;
  while (
      (j < num_pairs) &&
      (geox_data[j].delta_cost - geox_data[i - 1].delta_cost < kTieBreaker)) {
    j++;
  }
  return j;
}

std::vector<PairedDelta> GetPairedDeltaSorted(
    const std::vector<GeoPairValues>& geox_data) {
  const int num_pairs = static_cast<int>(geox_data.size());
  std::vector<PairedDelta> paired_delta_sorted(
      static_cast<int>((0.5 * num_pairs) * (num_pairs - 1)));

  // Creates paired_delta_sorted_.
  int iter = 0;
  for (int i = 1; i < num_pairs; ++i)
    for (int j = 0; j < i; ++j) {
      const double delta = GeoxDataUtil::CalculateRatio(
          geox_data[i].delta_response - geox_data[j].delta_response,
          geox_data[i].delta_cost - geox_data[j].delta_cost);
      paired_delta_sorted[iter] = {i, j, delta};
      iter++;
    }

  std::sort(paired_delta_sorted.begin(), paired_delta_sorted.end(),
            [&](const PairedDelta& a, const PairedDelta& b) {
              return (a.delta < b.delta);
            });

  return paired_delta_sorted;
}

}  // namespace trimmedmatch
