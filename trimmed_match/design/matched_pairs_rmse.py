# Copyright 2020 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""Module to evaluate root mean square error (RMSE) from matched pairs."""
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd

from trimmed_match import estimator
from trimmed_match.design import common_classes

GeoLevelData = common_classes.GeoLevelData
GeoXType = common_classes.GeoXType
GeoLevelPotentialOutcomes = common_classes.GeoLevelPotentialOutcomes

# Not much trimming (default to 10%) is needed if pairs are matched well, in
# other words, RMSE may be high otherwise.
_MAX_TRIM_RATE_FOR_RMSE_EVAL = 0.10


def _construct_potential_outcomes(
    geox_type: GeoXType, geo_eval_data: pd.DataFrame,
    incremental_spend_ratio: float,
    hypothesized_iroas: float) -> Dict[int, GeoLevelPotentialOutcomes]:
  """Construct two potential outcomes for each geo based on GeoXType.

  Assumes that incremental spend is proportional to geo_eval_data.spend.

  Args:
    geox_type: GeoXType.
    geo_eval_data: pd.DataFrame with columns (geo, response, spend).
    incremental_spend_ratio: float, the value multiplied by pretest spend
      to obtain incremental spend.
    hypothesized_iroas: float.

  Returns:
    Dict[int, GeoLevelPotentialOutcomes], map from geo to potential outcomes.
  """
  potential_outcomes = {}
  for _, row in geo_eval_data.iterrows():
    spend_in_control = row.spend
    incremental_spend = row.spend * incremental_spend_ratio
    if geox_type == GeoXType.GO_DARK:
      # Sometimes we use a spend proxy, which may have a different scale, so
      # the level of spend in control should be rescaled according to
      # incremental_spend_ratio.
      spend_in_control = incremental_spend
      incremental_spend = -incremental_spend
      spend_in_treatment = 0.0
    elif geox_type == GeoXType.HOLD_BACK:
      # Recall that a proxy cost is used for HOLD_BACK.
      spend_in_control = 0.0
      spend_in_treatment = incremental_spend
    elif geox_type == GeoXType.HEAVY_UP:
      spend_in_treatment = row.spend + incremental_spend
    elif geox_type == GeoXType.HEAVY_DOWN:
      incremental_spend = -row.spend * min(1.0, incremental_spend_ratio)
      spend_in_treatment = row.spend + incremental_spend
    else:
      raise ValueError("Unknown geox_type: {!r}".format(geox_type))

    controlled_outcome = GeoLevelData(row.geo, row.response, spend_in_control)
    treated_outcome = GeoLevelData(
        row.geo, max(0.0,
                     row.response + hypothesized_iroas * incremental_spend),
        spend_in_treatment)

    potential_outcomes[row.geo] = GeoLevelPotentialOutcomes(
        controlled_outcome, treated_outcome)

  return potential_outcomes


def _is_paired(ordered_list: List[int]) -> bool:
  """Check whether values are paired in ordered_list."""
  consecutive_comparison = [
      t == s for s, t in zip(ordered_list, ordered_list[1:])
  ]
  return (all(consecutive_comparison[0::2]) &
          (not any(consecutive_comparison[1::2])))


class MatchedPairsRMSE(object):
  """Evaluates RMSE for a randomized design based on a set of matched geo pairs.

  Example usage:
    geox_type = GeoXType.GO_DARK
    geo_pairs_eval_data = pd.DataFrame({
        "geo": [1, 2, 3, 4, 5, 6],
        "pair": [1, 1, 2, 2, 3, 3],
        "response": [10, 20, 30, 40, 50, 60],
        "spend": [10, 20, 30, 40, 50, 60]
    })
    budget = 10
    hypothesized_iroas = 1
    mpr = MatchedPairsRMSE(geox_type, geo_pairs_eval_data, budget,
    hypothesized_iroas)
    rmse, detailed_result = mpr.report(num_simulations=1000, max_trim_rate=0.10)
  """

  def __init__(self,
               geox_type: GeoXType,
               geo_pairs_eval_data: pd.DataFrame,
               budget: float,
               hypothesized_iroas: float,
               base_seed: int = 0):
    """Initializes the class.

    Args:
      geox_type: GeoXType.
      geo_pairs_eval_data: pd.DataFrame(geo, pair, response, spend).
      budget: float.
      hypothesized_iroas: float >=0.
      base_seed: int, seed for generating random numbers.

    Raises:
      KeyError: geos are not unique or are not in pairs in geo_pairs_eval_data.
      ValueError: if hypothesized_iroas is negative or geos are not paired
        properly.
    """
    if len(set(geo_pairs_eval_data.geo)) != geo_pairs_eval_data.shape[0]:
      raise ValueError("Geos are not unique in geo_pairs_eval_data")

    if hypothesized_iroas < 0:
      raise ValueError(f"iROAS must be positive, got {hypothesized_iroas}")

    sorted_geo_pairs_eval_data = geo_pairs_eval_data.sort_values(
        by=["pair", "geo"])
    if not _is_paired(list(sorted_geo_pairs_eval_data.pair)):
      raise KeyError("Geos in geo_pairs_eval_data are not paired properly")

    self._budget = budget
    self._hypothesized_iroas = hypothesized_iroas
    self._base_seed = base_seed
    self._potential_outcomes = _construct_potential_outcomes(
        geox_type, sorted_geo_pairs_eval_data,
        (budget * 2.0 / sorted_geo_pairs_eval_data.spend.sum()),
        hypothesized_iroas)
    self._paired_geos = {}
    for _, row in sorted_geo_pairs_eval_data.iterrows():
      if row.pair not in self._paired_geos:
        self._paired_geos[row.pair] = {"left": row.geo}
      else:
        self._paired_geos[row.pair]["right"] = row.geo

  def _simulate_geox_data(self, i: int) -> Dict[int, GeoLevelPotentialOutcomes]:
    """Generates a random geo assignment and geox data from matched pairs.

    Args:
      i: int, random seed adding to self._base_seed.

    Returns:
      Dict, which maps pair to (data for control geo, data for treatment geo)].
    """
    np.random.seed(i + self._base_seed)
    assignments = np.sign(
        np.random.uniform(low=-1, high=1, size=len(self._paired_geos)))
    geox_data = {}
    index = 0
    for pair, geos in self._paired_geos.items():
      if assignments[index] > 0:
        [control_geo, treatment_geo] = [geos["left"], geos["right"]]
      else:
        [control_geo, treatment_geo] = [geos["right"], geos["left"]]
      control_geo_potential_outcomes = self._potential_outcomes[control_geo]
      treatment_geo_potential_outcomes = self._potential_outcomes[treatment_geo]
      geox_data[pair] = GeoLevelPotentialOutcomes(
          control_geo_potential_outcomes.controlled,
          treatment_geo_potential_outcomes.treated)
      index += 1
    return geox_data

  def report(self,
             num_simulations: int = 1000,
             max_trim_rate: float = _MAX_TRIM_RATE_FOR_RMSE_EVAL,
             trim_rate: float = -1.0) -> Tuple[float, pd.DataFrame]:
    """Reports the RMSE.

    Args:
      num_simulations: int.
      max_trim_rate: float.
      trim_rate: float, with default (-1.0) trim_rate is data-driven.

    Returns:
      RMSE: rmse of the iROAS estimate obtained from multiple simulations.
      detailed_results: a list of estimator.TrimmedMatch elements, one for each
        simulation. Each element contains the fields: estimate, std_error,
        conf_interval_low, conf_interval_up, trim_rate, ci_level.

    Raises:
      ValueError: if trim_rate is greater than max_trim_rate or
                  if max_trim_rate is outside [0, 0.5].
    """
    if trim_rate > max_trim_rate:
      raise ValueError(f"trim_rate {trim_rate} is greater than max_trim_rate "
                       f"which is {max_trim_rate}.")
    if max_trim_rate < 0 or max_trim_rate >= 0.5:
      raise ValueError("max_trim_rate must be in [0, 0.5), but got "
                       f"{max_trim_rate}.")

    point_estimates = np.zeros(num_simulations)
    detailed_results = []
    for index in range(num_simulations):
      geox_data = self._simulate_geox_data(index)
      pairs = list(geox_data.keys())
      delta_response = [
          geox_data[pair].treated.response - geox_data[pair].controlled.response
          for pair in pairs
      ]
      delta_spend = [
          geox_data[pair].treated.spend - geox_data[pair].controlled.spend
          for pair in pairs
      ]
      fit = estimator.TrimmedMatch(delta_response, delta_spend, max_trim_rate)
      report = fit.Report(trim_rate=trim_rate)
      trimmed_pairs = [pairs[i] for i in report.trimmed_pairs_indices]
      detailed_results.append({
          "simulation": index,
          "estimate": report.estimate,
          "std_error": report.std_error,
          "trim_rate": report.trim_rate,
          "ci_level": report.confidence,
          "conf_interval_low": report.conf_interval_low,
          "conf_interval_up": report.conf_interval_up,
          "trimmed_pairs": trimmed_pairs,
      })
      point_estimates[index] = report.estimate
    rmse = np.sqrt(np.mean((point_estimates - self._hypothesized_iroas)**2))
    detailed_results = pd.DataFrame(detailed_results)

    return rmse, detailed_results
