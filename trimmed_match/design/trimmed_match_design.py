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
"""Library to create a matched pairs design for randomized geo experiment.

Example usage:
  import pandas as pd
  geox_type = GeoXType.HEAVY_UP
  pretest_data = pd.DataFrame(data={
    'date': ['2019-01-01'] * 20 + ['2019-03-01'] * 20,
    'geo': [1, 2] * 20,
    'sales': range(40),
    'cost': range(40),
    'transactions': range(40)})
  response = 'sales'
  spend_proxy = 'cost'
  matching_metrics = {'sales': 1.0, 'transactions': 1.0, 'cost': 0.01}
  time_window_for_design = ['2019-01-01', '2019-03-01']
  time_window_for_eval = ['2019-02-01', '2019-03-01']

  # Create candidate designs
  tmd = trimmed_match_design.TrimmedMatchGeoXDesign(
    geox_type, pretest_data, response, spend_proxy, matching_metrics,
    time_window_for_design, time_window_for_eval)
  budget_list = [1.0, 2.0]
  iroas_list = [0.0, 1.0, 2.0, 3.0]
  candidate_designs = tmd.report_candidate_designs(budget_list, iroas_list)
"""

from typing import Dict, List, Optional, Tuple
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from trimmed_match.design import common_classes
from trimmed_match.design import geo_assignment
from trimmed_match.design import geo_level_estimators
from trimmed_match.design import matched_pairs_rmse
from trimmed_match.design import plot_utilities
from trimmed_match.design import util

_DEFAULT_CONFIDENCE_LEVEL = 0.8
# Default parameters for the assignment tests
_DEFAULT_MIN_RESPONSE_SHARE = 0.0
_DEFAULT_MIN_NUMBER_OF_GEOS_PER_GROUP = 1
_DEFAULT_MAX_FRACTION_LARGEST_GEO = 1.0
# Minimum number of geo pairs
_MIN_NUM_PAIRS = 10
# Minimum total spend
_SPEND_TOLERANCE = 1e-10
# Minimum acceptable value for iROAS
_MIN_IROAS = 0

TimeWindow = common_classes.TimeWindow
GeoXType = common_classes.GeoXType
MatchedPairsRMSE = matched_pairs_rmse.MatchedPairsRMSE
GeoXEstimator = geo_level_estimators.GeoXEstimator

PRE_EXPERIMENT = common_classes.ExperimentPeriod.PRE_EXPERIMENT
EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT


class TrimmedMatchGeoXDesign:
  """A randomized geo experimental design using Trimmed Match."""

  def __init__(self,
               geox_type: GeoXType,
               pretest_data: pd.DataFrame,
               time_window_for_design: TimeWindow,
               time_window_for_eval: TimeWindow,
               response: str = 'response',
               spend_proxy: str = 'spend',
               matching_metrics: Optional[Dict[str, float]] = None,
               pairs: Optional[List[pd.DataFrame]] = None):
    """Initializes the instance.

    Args:
      geox_type: str, type of the experiment. See supported values in GeoXType.
      pretest_data: pd.DataFrame (date, geo, ...).
      time_window_for_design: TimeWindow, representing the time period of
        pretest data used for the design (training + eval).
      time_window_for_eval: TimeWindow, representing the time period of pretest
        data used for evaluation of RMSE in estimating iROAS.
      response: str, a column name in pretest_data, by default 'response'.
      spend_proxy: str, a column name in pretest_data, by default 'spend'.
      matching_metrics: dict, mapping a column name to a numeric weight.
      pairs: optional, list of dataframes with columns (geo1, geo2, pair)
        containing the pairs of geos to use for the power analysis.

    Raises:
      ValueError: response or spend_proxy is not in pretest_data.
      ValueError: number of geos in pretest_data is not even.
      ValueError: unable to convert either column response or spend_proxy to
                  numeric.
      ValueError: total spend_proxy is zero
    """
    if response not in pretest_data.columns:
      raise ValueError(f'{response} is not available in pretest_data.')
    if spend_proxy not in pretest_data.columns:
      raise ValueError(f'{spend_proxy} is not available in pretest_data.')
    if pretest_data[spend_proxy].sum() < _SPEND_TOLERANCE:
      raise ValueError(f'The column {spend_proxy} should have some positive ' +
                       f'entries. The sum of {spend_proxy} found is ' +
                       f'{pretest_data[spend_proxy].sum():.2f}')
    if matching_metrics is None:
      matching_metrics = {response: 1.0, spend_proxy: 0.01}
    if sum(matching_metrics.values()) <= 0:
      raise ValueError('Weights in matching_metrics sum up to 0.')

    self._geox_type = geox_type
    self._response = response
    self._spend_proxy = spend_proxy

    self._matching_metrics = matching_metrics.copy()
    if self._response not in self._matching_metrics:
      self._matching_metrics[self._response] = 0.0
    if self._spend_proxy not in self._matching_metrics:
      self._matching_metrics[self._spend_proxy] = 0.0

    self._pretest_data = util.check_input_data(
        pretest_data, list(self._matching_metrics.keys()))
    # Add period to pre-test data.
    training_and_evaluation = self._pretest_data['date'].between(
        time_window_for_design.first_day, time_window_for_design.last_day
    ) | self._pretest_data['date'].between(
        time_window_for_eval.first_day, time_window_for_eval.last_day
    )
    self._pretest_data = self._pretest_data[training_and_evaluation]
    self._pretest_data['period'] = np.where(
        self._pretest_data['date'].between(
            time_window_for_eval.first_day, time_window_for_eval.last_day
        ),
        EXPERIMENT,
        PRE_EXPERIMENT,
    )
    self._time_window_for_design = time_window_for_design
    self._time_window_for_eval = time_window_for_eval

    if pairs is not None:
      try:
        util.check_pairs(self._pretest_data, pairs)
      except ValueError as error:
        raise error
    # pairs, a dataframe with columns (geo1, geo2, pair) containing the
    # pairs of geos to use for the power analysis.
    self._pairs = pairs
    # geo_level_eval_data, a list of pd.DataFrame with columns (geo, response,
    # spend, pair) for evaluation of the RMSE.
    self._geo_level_eval_data = None
    self._geo_period_level_data = None

  @property
  def pairs(self):
    return self._pairs

  @property
  def geo_level_eval_data(self):
    return self._geo_level_eval_data

  @property
  def geo_period_level_data(self):
    return self._geo_period_level_data

  def _create_sign_test_data(self) -> pd.DataFrame:
    """Creates sign test data based on latest pretest data.

    Returns:
      pd.DataFrame with columns (geo, response, spend), where response and
      spend are geo-level overall values for the most recent time period of the
      same duration as self._time_window_for_eval.
    """
    available_dates = self._pretest_data['date'].drop_duplicates().sort_values(
        ascending=False)
    eval_duration = available_dates.between(
        self._time_window_for_eval.first_day,
        self._time_window_for_eval.last_day).sum()
    latest_data = self._pretest_data[self._pretest_data['date'].isin(
        available_dates[:eval_duration])]

    return latest_data.groupby(
        by='geo', as_index=False)[[self._response, self._spend_proxy]].sum()

  def create_geo_pairs(self, use_cross_validation: bool = True):
    """Creates geo pairs using pretest data and data to evaluate the RMSE.

    Args:
      use_cross_validation: bool, if True then geo pairing uses pretest data
        during time_window_for_design but not during time_window_for_eval,
        otherwise geo pairing uses pretest data during time_window_for_design
        and time_window_for_eval.
    """
    pretest = self._pretest_data.copy()
    for metric in self._matching_metrics:
      if use_cross_validation:
        pretest['training_' + metric] = (
            pretest['period'] == PRE_EXPERIMENT
        ) * pretest[metric]
      else:
        pretest['training_' + metric] = pretest[metric]

    pretest = (
        pretest.drop(columns=['date']).groupby('geo', as_index=False).sum()
    )
    # if the number of geos is odd, remove the largest geo for pairing
    if self._pretest_data['geo'].nunique() % 2 != 0:
      largest_geo = pretest.sort_values('response', ascending=False)[
          'geo'
      ].iloc[0]
      pretest = pretest[pretest['geo'] != largest_geo]

    pretest['rankscore'] = 0
    for metric, weight in self._matching_metrics.items():
      pretest['rankscore'] += weight * (
          pretest['training_' + metric] / sum(pretest['training_' + metric])
      )

    geos_ordered = pretest.sort_values(
        ['rankscore', 'geo'], ascending=[False, True]
    ).reset_index(drop=True)
    geopairs_left = geos_ordered.iloc[::2, :].reset_index(drop=True)
    geopairs_right = geos_ordered.iloc[1::2, :].reset_index(drop=True)

    # order by weighted distance between metrics
    dist = 0
    for metric, weight in self._matching_metrics.items():
      dist += weight * (
          abs(geopairs_left['training_' + metric] -
              geopairs_right['training_' + metric]) /
          sum(geos_ordered['training_' + metric]))

    pairs = (pd.DataFrame({
        'geo1': geopairs_left['geo'],
        'geo2': geopairs_right['geo'],
        'distance': dist
    }).sort_values(by=['distance', 'geo1'],
                   ascending=[False, True])).reset_index(drop=True)
    npairs = pairs.shape[0]
    pairs['pair'] = range(1, npairs + 1)

    self._pairs = [
        pairs[pairs['pair'] > x].reset_index(drop=True)
        for x in range(0, npairs)
    ]

  def create_geo_level_eval_data(self):
    """Creates geo level data to evaluate the RMSE.

    Create geo_level_eval_data, a pd.DataFrame with columns (geo, response,
    spend, pair) for evaluation of the RMSE for each pairing available.
    """
    if self.pairs is None:
      raise ValueError('pairs are not specified.')

    pretest_data = self._pretest_data.copy()
    pretest_data = pretest_data.groupby(['geo', 'period'], as_index=False)[
        [self._response, self._spend_proxy]
    ].sum()
    eval_data = pretest_data[pretest_data['period'] == EXPERIMENT].groupby(
        'geo', as_index=False)[[self._response, self._spend_proxy]].sum()

    geo_level_eval_data = []
    geo_period_level_data = []
    for pairing in self.pairs:
      pairing.sort_values(by='pair', inplace=True, ignore_index=True)
      geo_to_pair = pd.DataFrame({
          'geo':
              pairing['geo1'].tolist() +
              pairing['geo2'].tolist(),
          'pair':
              pairing['pair'].tolist() +
              pairing['pair'].tolist()
      })
      geo_level_eval_data.append(pd.merge(
          eval_data[['geo', self._response, self._spend_proxy]],
          geo_to_pair,
          on='geo').rename(columns={
              self._response: 'response',
              self._spend_proxy: 'spend'
          }).sort_values(by=['pair', 'geo']).reset_index(drop=True))

      geo_period_level_data.append(pd.merge(
          pretest_data[['geo', 'period', self._response, self._spend_proxy]],
          geo_to_pair,
          on='geo').rename(columns={
              self._response: 'response',
              self._spend_proxy: 'spend'
          }).sort_values(by=['pair', 'geo']).reset_index(drop=True))

    self._geo_level_eval_data = geo_level_eval_data
    self._geo_period_level_data = geo_period_level_data

  def report_candidate_designs(
      self,
      budget_list: List[float],
      iroas_list: List[float],
      use_cross_validation: bool = True,
      num_simulations: int = 200,
      max_trim_rate: float = 0.10,
      estimator: GeoXEstimator = geo_level_estimators.TrimmedMatch(),
  ) -> Tuple[pd.DataFrame, Dict[Tuple[float, float, int], pd.DataFrame]]:
    """Report the RMSE of iROAS estimate and summary for each candidate design.

    Args:
      budget_list: list of floats.
      iroas_list: list of nonnegative floats.
      use_cross_validation: bool, same as in create_geo_pairs().
      num_simulations: int, num of simulations for RMSE evaluation.
      max_trim_rate: float, the argument for estimator.TrimmedMatch; a small
        value implies the need of less trimming, i.e. high quality pairs.
      estimator: GeoXEstimator, to use for the analysis.

    Returns:
      results: pd.DataFrame, with columns (num_pairs,
        experiment_response, experiment_spend, spend_response_ratio, budget,
        iroas, rmse, proportion_cost_in_experiment), where experiment_response
        and experiment_spend are the total response and total spend,
        respectively, for both treatment and control during the eval time
        window, and spend_response_ratio is the ratio of experiment_spend to
        experiment_response. Therefore, for hold-back (e.g. LC) or
        go-dark experiments, the cost/baseline ratio for the treatment group is
        equal to spend_response_ratio * 2.
      detailed_results: dict with keys (budget, iroas, pair_index) and
        values pd.DataFrames with the results of each simulation. The
        pd.DataFrames have columns (simulation, estimate, trim_rate,
        conf_interval_low, conf_interval_up, ci_level).

    Raises:
      ValueError if any element in iroas_list is negative.
      ValueError if all pairings don't have the minimum number of pairs.
      ValueError if no suitable pairing could be found.
    """
    if self.pairs is None:
      self.create_geo_pairs(use_cross_validation)

    self.create_geo_level_eval_data()

    num_pairs_list = [len(x.index) for x in self.pairs]

    not_recommended_pairings = [
        x for x in range(len(self.pairs)) if num_pairs_list[x] < _MIN_NUM_PAIRS
    ]
    warnings.warn(
        'We will not attempt to use the pairing in position '
        f'{not_recommended_pairings} as we recommend to have at least '
        f'{_MIN_NUM_PAIRS} pairs in the design.'
    )
    pairs_index_list = [
        x for x in range(len(self.pairs)) if num_pairs_list[x] >= _MIN_NUM_PAIRS
    ]
    if not pairs_index_list:
      raise ValueError(
          f'All pairings were rejected due to less than {_MIN_NUM_PAIRS} pairs '
          'in the design.'
      )

    if min(iroas_list) < _MIN_IROAS:
      invalid_iroas = [iroas for iroas in iroas_list if iroas < _MIN_IROAS]
      raise ValueError(
          'All elements in iroas_list must have non-negative values, got '
          f'{invalid_iroas}.'
      )

    total_spend = self._pretest_data.loc[
        self._pretest_data['period'] == EXPERIMENT, self._spend_proxy
    ].sum()
    results = []
    detailed_results = {}
    for iroas in iroas_list:
      for budget in budget_list:
        for ind in pairs_index_list:
          if self.geo_level_eval_data[ind]['spend'].sum() < _SPEND_TOLERANCE:
            warnings.warn(
                'the total spend during the evaluation period for the pairing '
                f'in index {ind} is <{_SPEND_TOLERANCE}.'
            )
            continue

          if self.geo_level_eval_data[ind]['response'].sum() == 0:
            warnings.warn(
                'the total response during the evaluation period for the '
                f'pairing in index {ind} is 0.'
            )
            continue

          matched_rmse_class = MatchedPairsRMSE(self._geox_type,
                                                self.geo_period_level_data[ind],
                                                budget, iroas)
          (rmse, detailed_simulations) = matched_rmse_class.report(
              num_simulations=num_simulations,
              max_trim_rate=max_trim_rate,
              estimator=estimator)

          experiment_response = self.geo_level_eval_data[ind]['response'].sum()
          experiment_spend = self.geo_level_eval_data[ind]['spend'].sum()

          sqrt_n_sim = np.sqrt(num_simulations)
          # Compute the std of RMSE using the delta method.
          mse_std = (
              (detailed_simulations.estimate - iroas)**2
          ).std() / sqrt_n_sim
          rmse_std = mse_std / (2 * rmse)

          results.append({
              'pair_index':
                  ind,
              'num_pairs':
                  num_pairs_list[ind],
              'experiment_response':
                  experiment_response,
              'experiment_spend':
                  experiment_spend,
              'spend_response_ratio':
                  experiment_spend / experiment_response,
              'budget':
                  budget,
              'iroas':
                  iroas,
              'rmse':
                  rmse,
              'rmse_std':
                  rmse_std,
              'proportion_cost_in_experiment':
                  experiment_spend / total_spend,
              'rmse_cost_adjusted':
                  rmse / (experiment_spend / total_spend),
              'rmse_cost_adjusted_std':
                  rmse_std / (experiment_spend / total_spend),
              'cihw': (detailed_simulations['conf_interval_up'] -
                       detailed_simulations['conf_interval_low']).mean() / 2,
              'std':
                  detailed_simulations.estimate.std(),
              'bias': (detailed_simulations.estimate - iroas).mean(),
              'bias_std':
                  (detailed_simulations.estimate - iroas).std() / sqrt_n_sim,
              'trim_rate':
                  detailed_simulations.trim_rate.mean(),
              'trim_rate_std':
                  detailed_simulations.trim_rate.std() / sqrt_n_sim,
              'trim_rate_cost':
                  detailed_simulations.trim_rate_cost.mean(),
              'trim_rate_cost_std':
                  detailed_simulations.trim_rate_cost.std() / sqrt_n_sim,
          })
          detailed_results[(budget, iroas, ind)] = detailed_simulations

    if not results:
      raise ValueError('All the pairings that were tested had total spend '
                       f'<{_SPEND_TOLERANCE} or total response equal to 0 '
                       'during the evaluation period.')

    results = pd.DataFrame(results)
    return (results, detailed_results)

  def plot_candidate_design(
      self, results: pd.DataFrame) -> Dict[Tuple[float, float], plt.Axes]:
    """Plot the RMSE curve for a set of candidate designs.

    Args:
      results: pd.DataFrame, with columns (pair_index, experiment_response,
        experiment_spend, spend_response_ratio, budget,
        iroas, rmse, proportion_cost_in_experiment). Results can be the output
        of the method report_candidate_designs.

    Returns:
      axes_dict: a dictionary with keys (budget, iroas) with the plot of the
        RMSE values as a function of the number of excluded pairs for the design
        with corresponding budget and iROAS.
    """
    axes_dict = plot_utilities.plot_candidate_design_rmse(
        self._response, int(self._pretest_data['geo'].nunique() / 2), results)

    return axes_dict

  def output_chosen_design(
      self,
      pair_index: int,
      base_seed: int,
      confidence: float = _DEFAULT_CONFIDENCE_LEVEL,
      group_control: int = common_classes.GeoAssignment.CONTROL,
      group_treatment: int = common_classes.GeoAssignment.TREATMENT
  ) -> np.ndarray:
    """Plot the comparison between treatment and control of a candidate design.

    Args:
      pair_index: int, index of the pairing chosen for the experiment.
      base_seed: seed for the random number generator.
      confidence: float in (0, 1), confidence level for 2-sided CI.
      group_control: value representing the control group in the data.
      group_treatment: value representing the treatment group in the data.

    Returns:
      an array of subplots containing the scatterplot and time series comparison
        for the response and spend of the two groups.
    """
    self.generate_balanced_assignment(
        pair_index=pair_index,
        base_seed=base_seed,
        confidence=confidence,
        group_control=group_control,
        group_treatment=group_treatment)

    return plot_utilities.output_chosen_design(
        self._pretest_data, self.geo_level_eval_data[pair_index],
        self._response, self._spend_proxy, self._time_window_for_eval,
        group_control, group_treatment)

  def generate_balanced_assignment(
      self,
      pair_index: int,
      base_seed: int,
      confidence: float = _DEFAULT_CONFIDENCE_LEVEL,
      min_response_share: float = _DEFAULT_MIN_RESPONSE_SHARE,
      min_number_of_geos_per_group: int = _DEFAULT_MIN_NUMBER_OF_GEOS_PER_GROUP,
      max_fraction_largest_geo: float = _DEFAULT_MAX_FRACTION_LARGEST_GEO,
      group_control: int = common_classes.GeoAssignment.CONTROL,
      group_treatment: int = common_classes.GeoAssignment.TREATMENT
  ):
    """Generate balanced assignment for the chosen candidate design.

    Args:
      pair_index: int, index of the pairing chosen for the experiment.
      base_seed: seed for the random number generator.
      confidence: float in (0, 1), confidence level for 2-sided CI.
      min_response_share: minimum response volume share for both groups as a
        percentage of the total response.
      min_number_of_geos_per_group: the minimum number of geos in each group.
      max_fraction_largest_geo: the maximum share of response covered by the
        largest geo in each group.
      group_control: value representing the control group in the data.
      group_treatment: value representing the treatment group in the data.
    """
    sign_data = self._create_sign_test_data().rename(
        columns={self._response: 'response', self._spend_proxy: 'spend'}
    )
    sign_test_data = pd.merge(
        sign_data,
        self._geo_level_eval_data[pair_index][['geo', 'pair']],
        on='geo',
        how='outer',
    ).fillna({'response': 0, 'spend': 0})
    sign_test_data = sign_test_data[~sign_test_data['pair'].isna()]
    sign_test_data['pair'] = sign_test_data['pair'].astype(int)
    aa_test_data = self._geo_level_eval_data[pair_index].copy()
    representativeness_test_data = self._geo_period_level_data[pair_index][
        ['geo', 'pair', 'response']
    ].groupby(
        ['geo', 'pair'], as_index=False
    ).sum()
    total_pretest_response = self._pretest_data['response'].sum()

    # TODO(b/282119678): replace np.random.seed by a random number generator.
    np.random.seed(base_seed)
    assignment = geo_assignment.generate_balanced_random_assignment(
        representativeness_test_data,
        sign_test_data,
        aa_test_data,
        confidence,
        confidence,
        min_response_share * total_pretest_response,
        min_number_of_geos_per_group,
        max_fraction_largest_geo,
    )
    self._geo_level_eval_data[pair_index] = self._geo_level_eval_data[
        pair_index
    ][['geo', 'pair', 'response', 'spend']].merge(
        assignment, on=['geo', 'pair'], how='left'
    )

    self._geo_level_eval_data[pair_index]['assignment'] = (
        self._geo_level_eval_data[pair_index]['assignment'].map(
            {False: group_control, True: group_treatment}
        )
    )

  def plot_pair_by_pair_comparison(
      self,
      pair_index: int,
      group_control: int = common_classes.GeoAssignment.CONTROL,
      group_treatment: int = common_classes.GeoAssignment.TREATMENT,
  ) -> sns.FacetGrid:
    """Plot the time series of the response variable for each pair.

    Args:
      pair_index: index of the pairing chosen for the experiment.
      group_control: value representing the control group in the data.
      group_treatment: value representing the treatment group in the data.

    Returns:
      g: sns.FacetGrid containing one axis for each pair of geos. Each axis
        contains the time series plot of the response variable for the
        treated geo vs the control geo for a particular pair in the design.
    """
    return plot_utilities.plot_paired_comparison(
        self._pretest_data,
        self._geo_level_eval_data[pair_index],
        self._response,
        self._time_window_for_design,
        self._time_window_for_eval,
        group_control,
        group_treatment,
    )
