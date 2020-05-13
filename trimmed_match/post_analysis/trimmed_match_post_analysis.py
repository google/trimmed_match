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

"""Utilities functions to report the experiment results in the colab.
"""

import collections
import pandas as pd
from trimmed_match import estimator
from trimmed_match.design import plot_utilities

TrimmedMatchData = collections.namedtuple('TrimmedMatchData', [
    'pair', 'treatment_response', 'control_response', 'treatment_cost',
    'control_cost', 'epsilon'
])
TrimmedMatchResults = collections.namedtuple('TrimmedMatchResults', [
    'data', 'report', 'trimmed_pairs', 'incremental_cost', 'lift',
    'treatment_response'
])


def prepare_data_for_post_analysis(geox_data: pd.DataFrame,
                                   exclude_cooldown: bool = True
                                  ) -> TrimmedMatchData:
  """Returns a data frame to be analysed using Trimmed Match.

  Args:
    geox_data: data frame with columns (geo, response, cost, period, pair,
      assignment)
      where period is 0 (pretest), 1 (test), 2 (cooldown) and -1 (others).
    exclude_cooldown: TRUE (only using test period), FALSE (using test +
      cooldown).

  Returns:
    dt: namedtuple with fields pair, treatment_response, treatment_cost,
      control_response, control_cost.
  """
  if exclude_cooldown:
    geo_data = geox_data[geox_data['period'] == 1].copy()
  else:
    geo_data = geox_data[geox_data['period'].isin([1, 2])].copy()

  grouped_data = geo_data.groupby(['pair', 'assignment'],
                                  as_index=False)['response', 'cost'].sum()

  grouped_data.sort_values(by='pair', inplace=True)

  is_treat = grouped_data['assignment'] == 1
  is_control = grouped_data['assignment'] == 0

  dt = TrimmedMatchData(
      pair=grouped_data.loc[is_treat, 'pair'].to_list(),
      treatment_response=grouped_data.loc[is_treat, 'response'].to_list(),
      control_response=grouped_data.loc[is_control, 'response'].to_list(),
      treatment_cost=grouped_data.loc[is_treat, 'cost'].to_list(),
      control_cost=grouped_data.loc[is_control, 'cost'].to_list(),
      epsilon=[],
  )
  return dt


def calculate_experiment_results(
    data: TrimmedMatchData,
    max_trim_rate: float = 0.25,
    confidence: float = 0.80,
    trim_rate: float = -1.0) -> TrimmedMatchResults:
  """Calculate the results of an experiment with Trimmed Match.

  Args:
    data: namedtuple with fields pair, treatment_response, treatment_cost,
      control_response, control_cost which can be obtained as output of the
      method prepare_data_for_post_analysis.
    max_trim_rate: half the largest fraction of pairs that can be
      trimmed.
    confidence: the confidence level for the two-sided confidence
      interval.
    trim_rate: trim rate, a value outside [0, max_trim_rate) triggers
      the data-driven choice described in the Trimmed Match paper.

  Returns:
    results: namedtuple with fields data, report, trimmed_pairs,
      incremental_cost, lift, treatment_response.
  """
  delta_response = [
      data.treatment_response[x] - data.control_response[x]
      for x in range(len(data.treatment_response))
  ]
  delta_spend = [
      data.treatment_cost[x] - data.control_cost[x]
      for x in range(len(data.treatment_response))
  ]
  tm = estimator.TrimmedMatch(delta_response, delta_spend, max_trim_rate)
  fit = tm.Report(confidence, trim_rate)

  trimmed_pairs = [data.pair[x] for x in fit.trimmed_pairs_indices]
  increm_cost = sum(data.treatment_cost) - sum(data.control_cost)
  lift = fit.estimate * increm_cost
  treatment_response = sum(data.treatment_response)
  epsilon = fit.epsilons
  data_updated = data._replace(epsilon=epsilon)

  results = TrimmedMatchResults(
      data=data_updated,
      report=fit,
      trimmed_pairs=trimmed_pairs,
      incremental_cost=increm_cost,
      lift=lift,
      treatment_response=treatment_response,
  )
  return results


def report_experiment_results(results: TrimmedMatchResults):
  """Report the results of Trimmed Match.

  Args:
    results: output of the function calculate_experiment_results with fields
      data, report, trimmed_pairs, incremental_cost, lift, treatment_response.

  """
  fit = results.report
  print('Summary of the results for the iROAS:\n\n')
  print('estimate\t std. error\t trim_rate\t ci_level\t confidence interval')
  print(
      '{:.3f}\t\t {:.3f}\t\t {:.2f}\t\t {:.2f}\t\t [{:.3f}, {:.3f}]\n\n'.format(
          fit.estimate, fit.std_error, fit.trim_rate, fit.confidence,
          fit.conf_interval_low, fit.conf_interval_up))

  print('cost = {}'.format(
      plot_utilities.human_readable_number(results.incremental_cost)))
  print('\nincremental response = {}'.format(
      plot_utilities.human_readable_number(results.lift)))
  print('\ntreatment response = {}'.format(
      plot_utilities.human_readable_number(results.treatment_response)))
  print('\nincremental response as % of treatment response = {:.2f}%\n'.format(
      results.lift * 100 / results.treatment_response))
