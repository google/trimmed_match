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
import copy
from typing import Tuple

import numpy as np
import pandas as pd
from trimmed_match.design import common_classes
from trimmed_match.design import geo_level_estimators


GeoXType = common_classes.GeoXType
GeoXEstimator = geo_level_estimators.GeoXEstimator

# Not much trimming (default to 10%) is needed if pairs are matched well, in
# other words, RMSE may be high otherwise.
_MAX_TRIM_RATE_FOR_RMSE_EVAL = 0.10
EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT
CONTROL = common_classes.GeoAssignment.CONTROL
TREATMENT = common_classes.GeoAssignment.TREATMENT
PRE_EXPERIMENT = common_classes.ExperimentPeriod.PRE_EXPERIMENT
EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT


def _construct_potential_outcomes(
    geox_type: GeoXType, geo_eval_data: pd.DataFrame,
    incremental_spend_ratio: float,
    hypothesized_iroas: float) -> pd.DataFrame:
  """Construct two potential outcomes for each geo based on GeoXType.

  Assumes that incremental spend is proportional to geo_eval_data.spend.

  Args:
    geox_type: GeoXType.
    geo_eval_data: pd.DataFrame with columns (geo, response, spend).
    incremental_spend_ratio: float, the value multiplied by pretest spend
      to obtain incremental spend.
    hypothesized_iroas: float.

  Returns:
    pd.DataFrame, dataframe with potential outcomes for spend and response.
  """
  potential_outcomes = geo_eval_data[["pair", "geo"]].copy()
  spend_change_in_control = 0.0
  spend_change_in_treatment = (geo_eval_data["spend"] * incremental_spend_ratio)

  # Compute spend potential outcomes
  # TODO(aiyouchen) add a check and raise an error if the spend change is too
  # large compared to BAU. In such case we should use
  # GO_DARK_TREATMENT_NOT_BAU_CONTROL
  if geox_type == GeoXType.GO_DARK:
    # this case corresponds to the standard GO_DARK, where control is in BAU
    # while treatment spend is shut down.
    potential_outcomes["spend_control"] = spend_change_in_treatment
    spend_change_in_treatment = -spend_change_in_treatment
    potential_outcomes["spend_treatment"] = 0.0
  elif geox_type == GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL:
    # this case corresponds to an experiment where treatment spend is shut
    # down, while control spend is changed (increased/decreased). Since
    # control is not in BAU, the potential outcomes for response when
    # controlled should be adjusted from the observed evaluation response.
    potential_outcomes["spend_control"] = spend_change_in_treatment
    spend_change_in_control = (potential_outcomes["spend_control"] -
                               geo_eval_data["spend"])
    spend_change_in_treatment = -geo_eval_data["spend"]
    potential_outcomes["spend_treatment"] = 0.0
  elif geox_type == GeoXType.HOLD_BACK:
    # Recall that a proxy cost is used for HOLD_BACK.
    potential_outcomes["spend_control"] = 0.0
    potential_outcomes["spend_treatment"] = spend_change_in_treatment
  elif geox_type == GeoXType.HEAVY_UP:
    potential_outcomes["spend_control"] = geo_eval_data["spend"]
    potential_outcomes["spend_treatment"] = (geo_eval_data["spend"] +
                                             spend_change_in_treatment)
  elif geox_type == GeoXType.HEAVY_DOWN:
    potential_outcomes["spend_control"] = geo_eval_data["spend"]
    spend_change_in_treatment = - (
        geo_eval_data["spend"] * min(1, incremental_spend_ratio))
    potential_outcomes["spend_treatment"] = (geo_eval_data["spend"] +
                                             spend_change_in_treatment)
  else:
    raise ValueError("Unknown geox_type: {!r}".format(geox_type))

  # Compute response potential outcomes
  potential_outcomes["response_control"] = np.maximum(
      0, geo_eval_data["response"] +
      hypothesized_iroas * spend_change_in_control)
  potential_outcomes["response_treatment"] = np.maximum(
      0, geo_eval_data["response"] +
      hypothesized_iroas * spend_change_in_treatment)
  return potential_outcomes


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
      geo_pairs_eval_data: pd.DataFrame(geo, pair, response, spend). The columns
        date and period are optional
      budget: float.
      hypothesized_iroas: float >=0.
      base_seed: int, seed for generating random numbers.

    Raises:
      KeyError: geos are not unique or are not in pairs in geo_pairs_eval_data.
      ValueError: if hypothesized_iroas is negative or geos are not paired
        properly.
    """
    required_columns = {"pair", "geo", "response", "spend"}
    missing_columns = required_columns - set(geo_pairs_eval_data.columns)
    if missing_columns:
      raise ValueError(f"Missing {missing_columns} column in " +
                       "geo_pairs_eval_data")

    shape = geo_pairs_eval_data.geo.nunique()
    if "date" in geo_pairs_eval_data.columns:
      shape *= geo_pairs_eval_data.date.nunique()
    elif "period" in geo_pairs_eval_data.columns:
      shape *= geo_pairs_eval_data.period.nunique()
    if shape < geo_pairs_eval_data.shape[0]:
      raise ValueError("Duplicated values in geo_pairs_eval_data")
    elif shape > geo_pairs_eval_data.shape[0]:
      raise ValueError("Missing values in geo_pairs_eval_data")

    if "period" in geo_pairs_eval_data.columns:
      if EXPERIMENT not in geo_pairs_eval_data.period.values:
        raise ValueError(f"Missing experiment value {EXPERIMENT} in " +
                         "geo_pairs_eval_data.period")

    if hypothesized_iroas < 0:
      raise ValueError(f"iROAS must be positive, got {hypothesized_iroas}")

    geo_pairs_check = geo_pairs_eval_data.groupby("pair")["geo"].nunique()
    if any(geo_pairs_check.values != 2):
      raise KeyError("Geos in geo_pairs_eval_data are not paired properly")

    self._budget = budget
    self._hypothesized_iroas = hypothesized_iroas
    self._base_seed = base_seed

    # Precompute geox data
    optional_columns = {"date", "period"} & set(geo_pairs_eval_data.columns)
    self._geox_data = (
        geo_pairs_eval_data[required_columns | optional_columns].copy()
        .sort_values(["pair", "geo"], ignore_index=True))
    self._geox_data["assignment"] = CONTROL
    if "period" not in self._geox_data.columns:
      self._geox_data["period"] = EXPERIMENT

    # Save potential outcomes
    spend = self._geox_data[self._geox_data.period == EXPERIMENT].spend.sum()
    incremental_spend_ratio = (budget * 2.0 / spend)
    self._potential_outcomes = _construct_potential_outcomes(
        geox_type, self._geox_data, incremental_spend_ratio,
        hypothesized_iroas)
    self._geos = self._geox_data[["pair", "geo"]].copy(
        ).drop_duplicates().geo.values
    self._n_pairs = self._potential_outcomes.pair.nunique()
    self._pairs = self._potential_outcomes.pair.unique()

  def _simulate_geox_data(self, i: int) -> pd.DataFrame:
    """Generates a random geo assignment and geox data from matched pairs.

    Args:
      i: int, random seed adding to self._base_seed.

    Returns:
      pd.DataFrame, with treatment assignment and realized potential outcomes.
    """
    # Compute the assignment. self._geos must be ordered by pair!
    np.random.seed(i + self._base_seed)
    assignment = np.random.uniform(low=-1, high=1, size=self._n_pairs) > 0
    treated_geos = self._geos[[2*i for i in range(self._n_pairs)] + assignment]

    # Generate exhaustive and mutually exclusive groups: control and treated
    # (during the experiment), and pre_experiment (before the experiment)
    geox_data = self._geox_data.copy()
    geox_data.loc[geox_data.geo.isin(treated_geos), "assignment"] = TREATMENT
    control = ((geox_data.assignment.values == CONTROL) &
               (geox_data.period.values == EXPERIMENT))
    treated = ((geox_data.assignment.values == TREATMENT) &
               (geox_data.period.values == EXPERIMENT))
    pre_experiment = (geox_data.period.values == PRE_EXPERIMENT)

    # Compute realized outcomes, given the assignment and the period
    geox_data["response"] = (
        self._potential_outcomes.response_treatment.values * treated +
        self._potential_outcomes.response_control.values * control +
        geox_data.response.values * pre_experiment)
    geox_data["spend"] = (
        self._potential_outcomes.spend_treatment.values * treated +
        self._potential_outcomes.spend_control.values * control +
        geox_data.spend.values * pre_experiment)
    return geox_data

  def report(self,
             num_simulations: int = 1000,
             max_trim_rate: float = _MAX_TRIM_RATE_FOR_RMSE_EVAL,
             trim_rate: float = -1.0,
             estimator: GeoXEstimator = geo_level_estimators.TrimmedMatch(),
             ) -> Tuple[float, pd.DataFrame]:
    """Reports the RMSE.

    Args:
      num_simulations: int.
      max_trim_rate: float.
      trim_rate: float, with default (-1.0) trim_rate is data-driven.
      estimator: GeoXEstimator, to use for the analysis

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
    tm_estimator = copy.deepcopy(estimator)
    tm_estimator.max_trim_rate = max_trim_rate
    tm_estimator.trim_rate = trim_rate

    point_estimates = np.zeros(num_simulations)
    detailed_results = []
    for index in range(num_simulations):
      geox_data = self._simulate_geox_data(index)
      report = tm_estimator.analyze(
          geoxts=geox_data.rename(columns={"spend": "cost"}),
          group_control=CONTROL,
          group_treatment=TREATMENT)
      trimmed_pairs = (
          [] if report.trimmed_pairs_indices is None
          else [self._pairs[i] for i in report.trimmed_pairs_indices])
      detailed_results.append({
          "simulation": index,
          "estimate": report.point_estimate,
          "trim_rate": report.trim_rate,
          "trim_rate_cost": report.trim_rate_cost,
          "ci_level": report.confidence_interval.confidence_level,
          "conf_interval_low": report.confidence_interval.lower,
          "conf_interval_up": report.confidence_interval.upper,
          "trimmed_pairs": trimmed_pairs,
      })
      point_estimates[index] = report.point_estimate
    rmse = np.sqrt(np.mean((point_estimates - self._hypothesized_iroas)**2))
    detailed_results = pd.DataFrame(detailed_results)

    return rmse, detailed_results
