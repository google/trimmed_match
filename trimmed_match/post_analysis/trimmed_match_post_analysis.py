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
import itertools

import numpy as np
import pandas as pd
from trimmed_match import estimator
from trimmed_match.design import common_classes
from trimmed_match.design import util

TrimmedMatchData = collections.namedtuple('TrimmedMatchData', [
    'pair', 'treatment_response', 'control_response', 'treatment_cost',
    'control_cost', 'epsilon'
])
TrimmedMatchResults = collections.namedtuple('TrimmedMatchResults', [
    'data', 'report', 'trimmed_pairs', 'incremental_cost', 'lift',
    'treatment_response'
])


def prepare_data_for_post_analysis(
    geox_data: pd.DataFrame,
    exclude_cooldown: bool = True,
    group_control: int = common_classes.GeoAssignment.CONTROL,
    group_treatment: int = common_classes.GeoAssignment.TREATMENT
) -> TrimmedMatchData:
  """Returns a data frame to be analysed using Trimmed Match.

  Args:
    geox_data: data frame with columns (geo, response, cost, period, pair,
      assignment)
      where period is 0 (pretest), 1 (test), 2 (cooldown) and -1 (others).
    exclude_cooldown: True (only using test period), False (using test +
      cooldown).
    group_control: value representing the control group in the data.
    group_treatment: value representing the treatment group in the data.

  Returns:
    dt: namedtuple with fields pair, treatment_response, treatment_cost,
      control_response, control_cost.

  Raises:
    ValueError: if the number of control and treatment geos is different, if any
    geo is duplicated, or if any pair does have one geo per group.
  """
  if exclude_cooldown:
    experiment_data = geox_data[geox_data['period'] == 1].copy()
  else:
    experiment_data = geox_data[geox_data['period'].isin([1, 2])].copy()

  grouped_data = experiment_data.groupby(['pair', 'assignment', 'geo'],
                                         as_index=False)[['response',
                                                          'cost']].sum()
  # remove any assignment outside treatment/control
  grouped_data = grouped_data[grouped_data['assignment'].isin(
      [group_control, group_treatment])]
  grouped_data.sort_values(by=['pair', 'assignment'], inplace=True)

  if any(grouped_data['geo'].duplicated()):
    raise ValueError('Some geos are duplicated and appear in multiple pairs.')

  expected = pd.DataFrame(
      data=itertools.product(grouped_data.pair.unique(),
                             [group_control, group_treatment]),
      columns=['pair', 'assignment']).sort_values(by=['pair', 'assignment'])
  if not np.array_equal(grouped_data[['pair', 'assignment']], expected):
    raise ValueError('Some pairs do not have one geo for each group.')

  is_treat = grouped_data['assignment'] == group_treatment
  is_control = grouped_data['assignment'] == group_control

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


def report_experiment_results(results: TrimmedMatchResults,
                              average_order_value: float):
  """Report the results of Trimmed Match.

  Args:
    results: output of the function calculate_experiment_results with fields
      data, report, trimmed_pairs, incremental_cost, lift, treatment_response.
    average_order_value: value used to convert transactions/visits to the
      "sales scale".
  """
  fit = results.report
  print('Summary of the results for the iROAS:\n\n')
  print('estimate\t std. error\t trim_rate\t ci_level\t confidence interval')
  print(
      '{:.3f}\t\t {:.3f}\t\t {:.2f}\t\t {:.2f}\t\t [{:.3f}, {:.3f}]\n\n'.format(
          fit.estimate * average_order_value,
          fit.std_error * average_order_value, fit.trim_rate, fit.confidence,
          fit.conf_interval_low * average_order_value,
          fit.conf_interval_up * average_order_value))

  print('cost = {}'.format(
      util.human_readable_number(results.incremental_cost)))
  print('\nincremental response = {}'.format(
      util.human_readable_number(results.lift)))
  print('\ntreatment response = {}'.format(
      util.human_readable_number(results.treatment_response)))
  print('\nincremental response as % of treatment response = {:.2f}%\n'.format(
      results.lift * 100 / results.treatment_response))


def check_input_data(
    data: pd.DataFrame,
    group_control: int = common_classes.GeoAssignment.CONTROL,
    group_treatment: int = common_classes.GeoAssignment.TREATMENT
) -> pd.DataFrame:
  """Returns data to be analysed using Trimmed Match with data imputation.

  Args:
    data: data frame with columns (date, geo, response, cost, pair, assignment).
    group_control: value representing the control group in the data.
    group_treatment: value representing the treatment group in the data.

  Returns:
    geox_data: data frame with columns (date, geo, response, cost, pair,
      assignment) and imputed missing data.

  Raises:
    ValueError: if the number of control and treatment geos is different, if any
    geo is duplicated, if any pair does have one geo per group, or if one of the
    groups is missing.
  """
  mandatory_columns = set(
      ['date', 'geo', 'response', 'cost', 'pair', 'assignment'])
  if not mandatory_columns.issubset(data.columns):
    raise ValueError('The mandatory columns ' +
                     f'{mandatory_columns - set(data.columns)} are missing ' +
                     'from the input data.')

  if not set([group_treatment, group_control]).issubset(
      set(data['assignment'].unique())):
    raise ValueError('The data do not have observations for the two groups.' +
                     'Check the data and the values used to indicate the ' +
                     'assignments for treatment and control. The labels ' +
                     'found in the data in input are ' +
                     f'{list(data["assignment"].unique())}, and the expected ' +
                     f'labels are: Treatment={group_treatment}, ' +
                     f'Control={group_control}')

  grouped_data = data[['pair', 'assignment', 'geo']].drop_duplicates()

  # remove any assignment outside treatment/control
  grouped_data = grouped_data[grouped_data['assignment'].isin(
      [group_control, group_treatment])]

  if any(grouped_data['geo'].duplicated()):
    raise ValueError(
        'Some geos are duplicated and appear in multiple pairs or groups.')

  expected = pd.DataFrame(
      data=itertools.product(grouped_data.pair.unique(),
                             [group_control, group_treatment]),
      columns=['pair', 'assignment'])
  if not np.array_equal(expected.sort_values(by=['pair', 'assignment']),
                        grouped_data[['pair', 'assignment']].sort_values(
                            by=['pair', 'assignment'])):
    raise ValueError('Some pairs do not have one geo for each group.')

  geos_and_dates = pd.merge(
      data[['date']].drop_duplicates().assign(key=1),
      data[['geo', 'pair', 'assignment']].drop_duplicates().assign(key=1),
      on='key',
      how='outer').drop(
          'key', axis=1)
  geox_data = pd.merge(
      geos_and_dates,
      data,
      on=['date', 'geo', 'pair', 'assignment'],
      how='left').fillna({
          'response': 0.0,
          'cost': 0.0
      })

  return geox_data[geox_data['assignment'].isin(
      [group_control, group_treatment])]
