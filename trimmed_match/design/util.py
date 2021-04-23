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

"""Utilities functions to manipulate the data in the colab."""

import datetime
import itertools
import operator
from typing import List

import dataclasses
import numpy as np
import pandas as pd
import pandas.io.formats.style as style
from scipy import stats
from trimmed_match.design import common_classes

TimeWindow = common_classes.TimeWindow
FormatOptions = common_classes.FormatOptions

_operator_functions = {'>': operator.gt,
                       '<': operator.lt,
                       '<=': operator.le,
                       '>=': operator.ge,
                       '=': operator.eq,
                       '!=': operator.ne}

_inverse_op = {'<': '>', '<=': '>=', '>': '<', '>=': '<=', '=': '!='}


@dataclasses.dataclass
class CalculateMinDetectableIroas:
  """Class for the calculation of the minimum detectable iROAS.

  Hypothesis testing for H0: iROAS=0 vs H1: iROAS>=min_detectable_iroas based
  on one sample X which follows a normal distribution with mean iROAS (unknown)
  and standard deviation rmse (known).

    Typical usage example:

    calc_min_detectable_iroas = CalculateMinDetectableIroas(0.1, 0.9)
    min_detectable_iroas = calc_min_detectable_iroas.at(2.0)

  """

  # chance of rejecting H0 incorrectly when H0 holds.
  significance_level: float = 0.1

  # chance of rejecting H0 correctly when H1 holds.
  power_level: float = 0.9

  # minimum detectable iroas at rmse=1.
  rmse_multiplier: float = dataclasses.field(init=False)

  def __post_init__(self):
    """Calculates rmse_multiplier.

    Raises:
      ValueError: if significance_level or power_level is not in (0, 1).
    """
    if self.significance_level <= 0 or self.significance_level >= 1.0:
      raise ValueError('significance_level must be in (0, 1), but got '
                       f'{self.significance_level}.')
    if self.power_level <= 0 or self.power_level >= 1.0:
      raise ValueError('power_level must be in (0, 1), but got '
                       f'{self.power_level}.')
    self.rmse_multiplier = (
        stats.norm.ppf(self.power_level) +
        stats.norm.ppf(1 - self.significance_level))

  def at(self, rmse: float) -> float:
    """Calculates min_detectable_iroas at the specified rmse."""
    return rmse * self.rmse_multiplier


def find_days_to_exclude(
    dates_to_exclude: List[str]) -> List[TimeWindow]:
  """Returns a list of time windows to exclude from a list of days and weeks.

  Args:
    dates_to_exclude: a List of strings with format indicating a single day as
    '2020/01/01' (YYYY/MM/DD) or an entire time period as
    '2020/01/01 - 2020/02/01' (indicating start and end date of the time period)

  Returns:
    days_exclude: a List of TimeWindows obtained from the list in input.
  """
  days_exclude = []
  for x in dates_to_exclude:
    tmp = x.split('-')
    if len(tmp) == 1:
      try:
        days_exclude.append(
            TimeWindow(pd.Timestamp(tmp[0]), pd.Timestamp(tmp[0])))
      except ValueError:
        raise ValueError(f'Cannot convert the string {tmp[0]} to a valid date.')
    elif len(tmp) == 2:
      try:
        days_exclude.append(
            TimeWindow(pd.Timestamp(tmp[0]), pd.Timestamp(tmp[1])))
      except ValueError:
        raise ValueError(
            f'Cannot convert the strings in {tmp} to a valid date.')
    else:
      raise ValueError(f'The input {tmp} cannot be interpreted as a single' +
                       ' day or a time window')

  return days_exclude


def expand_time_windows(periods: List[TimeWindow]) -> List[pd.Timestamp]:
  """Return a list of days to exclude from a list of TimeWindows.

  Args:
    periods: List of time windows (first day, last day).

  Returns:
    days_exclude: a List of obtained by expanding the list in input.
  """
  days_exclude = []
  for window in periods:
    days_exclude += pd.date_range(window.first_day, window.last_day, freq='D')

  return list(set(days_exclude))


def overlap_percent(dates_left: List['datetime.datetime'],
                    dates_right: List['datetime.datetime']) -> float:
  """Find the size of the intersections of two arrays, relative to the first array.

  Args:
    dates_left: List of datetime.datetime
    dates_right: List of datetime.datetime

  Returns:
    percentage: the percentage of elements of dates_right that also appear in
      dates_left
  """
  intersection = np.intersect1d(dates_left, dates_right)
  percentage = 100 * len(intersection) / len(dates_right)

  return percentage


def check_time_periods(geox_data: pd.DataFrame,
                       start_date_eval: pd.Timestamp,
                       start_date_aa_test: pd.Timestamp,
                       experiment_duration_weeks: int,
                       frequency: str) -> bool:
  """Checks that the geox_data contains the data for the two periods.

  Check that the geox_data contains all observations during the evaluation and
  AA test periods to guarantee that the experiment lasts exactly a certain
  number of days/weeks, depending on the frequency of the data (daily/weekly).

  Args:
    geox_data: pd.Dataframe with at least the columns (date, geo).
    start_date_eval: start date of the evaluation period.
    start_date_aa_test: start date of the aa test period.
    experiment_duration_weeks: int, length of the experiment in weeks.
    frequency: str indicating the frequency of the time series. It should be one
      of 'infer', 'D', 'W'.

  Returns:
    bool: a bool, True if the time periods specified pass all the checks

  Raises:
    ValueError: if part of the evaluation or AA test period are shorter than
      experiment_duration (either weeks or days).

  """
  if frequency not in ['infer', 'D', 'W']:
    raise ValueError(
        f'frequency should be one of ["infer", "D", "W"], got {frequency}')
  if frequency == 'infer':
    tmp = geox_data.copy().set_index(['date', 'geo'])
    frequency = infer_frequency(tmp, 'date', 'geo')

  if frequency == 'W':
    frequency = '7D'
    number_of_observations = experiment_duration_weeks
  else:
    number_of_observations = 7 * experiment_duration_weeks

  freq_str = 'weeks' if frequency == '7D' else 'days'

  missing_eval = find_missing_dates(geox_data, start_date_eval,
                                    experiment_duration_weeks,
                                    number_of_observations, frequency)
  if missing_eval:
    raise ValueError(
        (f'The evaluation period contains the following {freq_str} ' +
         f'{missing_eval} for which we do not have data.'))
  missing_aa_test = find_missing_dates(geox_data, start_date_aa_test,
                                       experiment_duration_weeks,
                                       number_of_observations, frequency)
  if missing_aa_test:
    raise ValueError((f'The AA test period contains the following {freq_str} ' +
                      f'{missing_aa_test} for which we do not have data.'))
  return True


def find_missing_dates(geox_data: pd.DataFrame, start_date: pd.Timestamp,
                       period_duration_weeks: int,
                       number_of_observations: int,
                       frequency: str) -> List[str]:
  """Find missing observations in a time period.

  Args:
    geox_data: pd.Dataframe with at least the columns (date, geo).
    start_date: start date of the evaluation period.
    period_duration_weeks: int, length of the period in weeks.
    number_of_observations: expected number of time points.
    frequency: str or pd.DateOffset indicating the frequency of the time series.

  Returns:
    missing: a list of strings, containing the dates for which data are missing
      in geox_data.
  """
  days = datetime.timedelta(days=7 * period_duration_weeks - 1)
  period_dates = ((geox_data['date'] >= start_date) &
                  (geox_data['date'] <= start_date + days))
  days_in_period = geox_data.loc[
      period_dates, 'date'].drop_duplicates().dt.strftime('%Y-%m-%d').to_list()
  missing = np.array([])
  if len(days_in_period) != number_of_observations:
    expected_observations = list(
        pd.date_range(start_date, start_date + days,
                      freq=frequency).strftime('%Y-%m-%d'))
    missing = set(expected_observations) - set(days_in_period)

  return sorted(missing)


def infer_frequency(data: pd.DataFrame, date_index: str,
                    series_index: str) -> str:
  """Infers frequency of data from pd.DataFrame with multiple indices.

  Infers frequency of data from pd.DataFrame with two indices, one for the slice
  name and one for the date-time.
  Example:
    df = pd.Dataframe{'date': [2020-10-10, 2020-10-11], 'geo': [1, 1],
      'response': [10, 20]}
    df.set_index(['geo', 'date'], inplace=True)
    infer_frequency(df, 'date', 'geo')

  Args:
    data: a pd.DataFrame for which frequency needs to be inferred.
    date_index: string containing the name of the time index.
    series_index: string containing the name of the series index.

  Returns:
    A str, either 'D' or 'W' indicating the most likely frequency inferred
    from the data.

  Raises:
    ValueError: if it is not possible to infer frequency of sampling from the
      provided pd.DataFrame.
  """
  data = data.sort_values(by=[date_index, series_index])
  # Infer most likely frequence for each series_index
  series_names = data.index.get_level_values(series_index).unique().tolist()
  series_frequencies = []
  for series in series_names:
    observed_times = data.iloc[data.index.get_level_values(series_index) ==
                               series].index.get_level_values(date_index)
    n_steps = len(observed_times)

    if n_steps > 1:
      time_diffs = (
          observed_times[1:n_steps] -
          observed_times[0:(n_steps - 1)]).astype('timedelta64[D]').values

      modal_frequency, _ = np.unique(time_diffs, return_counts=True)

      series_frequencies.append(modal_frequency[0])

  if not series_frequencies:
    raise ValueError(
        'At least one series with more than one observation must be provided.')

  if series_frequencies.count(series_frequencies[0]) != len(series_frequencies):
    raise ValueError(
        'The provided time series seem to have irregular frequencies.')

  try:
    frequency = {
        1: 'D',
        7: 'W'
    }[series_frequencies[0]]
  except KeyError:
    raise ValueError('Frequency could not be identified. Got %d days.' %
                     series_frequencies[0])

  return frequency


def human_readable_number(number: float) -> str:
  """Print a large number in a readable format.

  Return a readable format for a number, e.g. 123 milions becomes 123M.

  Args:
    number: a float to be printed in human readable format.

  Returns:
    readable_number: a string containing the formatted number.
  """
  number = float('{:.3g}'.format(number))
  magnitude = 0
  while abs(number) >= 1000 and magnitude < 4:
    magnitude += 1
    number /= 1000.0
  readable_number = '{}{}'.format('{:f}'.format(number).rstrip('0').rstrip('.'),
                                  ['', 'K', 'M', 'B', 'tn'][magnitude])
  return readable_number


def change_background_row(df: pd.DataFrame, value: float, operation: str,
                          column: str):
  """Colors a row of a table based on the expression in input.

  Color a row in:

    - orange if the value of the column satisfies the expression in input
    - beige if the value of the column satisfies the inverse expression in input
    - green otherwise

  For example, if the column has values [1, 2, 3] and we pass 'value' equal to
  2, and operation '>', then
   - 1 is marked in beige (1 < 2, which is the inverse expression)
   - 2 is marked in green (it's not > and it's not <)
   - 3 is marked in orange(3 > 2, which is the expression)

  Args:
    df: the table of which we want to change the background color.
    value: term of comparison to be used in the expression.
    operation: a string to define which operator to use, e.g. '>' or '='. For a
      full list check _operator_functions.
    column: name of the column to be used for the comparison

  Returns:
    pd.Series
  """
  if _operator_functions[operation](float(df[column]), value):
    return pd.Series('background-color: orange', df.index)

  elif _operator_functions[_inverse_op[operation]](float(df[column]), value):
    return pd.Series('background-color: beige', df.index)
  else:
    return pd.Series('background-color: lightgreen', df.index)


def flag_percentage_value(val, value: float, operation: str):
  """Colors a cell in red if its value satisfy the expression in input.

  Colors a cell in red if the expression is true for that cell, e.g. if the
  value of the cell is 10, 'value' in input is 5 and operation is '>', then we
  will color the cell in red as 10 > 5.

  Args:
    val: value in a cell of a dataframe.
    value: term of comparison used to decide the color of the cell.
    operation: a string to define which operator to use, e.g. '>' or '='. For a
      full list check _operator_functions.

  Returns:
    a str defining the color coding of the cell.
  """
  if _operator_functions[operation](float(val.strip(' %')), value):
    color = 'red'
  else:
    color = 'black'
  return 'color: %s' % color


def create_output_table(results: pd.DataFrame,
                        total_response: float,
                        total_spend: float,
                        geo_treatment: pd.DataFrame,
                        budgets_for_design: List[float],
                        average_order_value: float,
                        num_geos: int,
                        confidence_level: float = 0.9,
                        power_level: float = 0.8) -> pd.DataFrame:
  """Creates the table with the output designs.

  Args:
   results: table with columns (num_pairs_filtered,
     experiment_response, experiment_spend, spend_response_ratio, budget,
     iroas, rmse, proportion_cost_in_experiment) containing the generated
     design, e.g. the first output of the
     function TrimmedMatchGeoXDesign.report_candidate_design.
   total_response: total response for all geos (excluded as well) during the
     evaluation period.
   total_spend: total spend for all geos (excluded as well) during the
     evaluation period.
   geo_treatment: table with columns (geo, response, spend, pair) containing the
     treatment geos and their overall response and spend during the evaluation
     period.
   budgets_for_design: list of budgets to be considered for the designs.
   average_order_value: factor used to change scale from conversion count to
     conversion value.
   num_geos: number of geos available.
   confidence_level: confidence level for the test H0: iROAS=0
     vs H1: iROAS>=minimum_detectable_iroas.
   power_level: level used  for the power analysis.

  Returns:
    a pd.DataFrame with the output designs.
  """
  calc_min_detectable_iroas = CalculateMinDetectableIroas(
      1 - confidence_level, power_level)

  designs = []
  for budget in budgets_for_design:
    tmp_result = results[results['budget'] == budget]
    chosen_design = tmp_result.loc[tmp_result['rmse_cost_adjusted'].idxmin()]
    baseline = geo_treatment.loc[
        geo_treatment['pair'] > chosen_design['num_pairs_filtered'],
        'response'].sum()
    cost_in_experiment = geo_treatment.loc[
        geo_treatment['pair'] > chosen_design['num_pairs_filtered'],
        'spend'].sum()
    min_detectable_iroas_raw = calc_min_detectable_iroas.at(
        chosen_design['rmse'])
    min_detectable_iroas = average_order_value * min_detectable_iroas_raw
    min_detectable_lift = budget * 100 * min_detectable_iroas_raw / baseline
    num_removed_geos = int(2 * chosen_design['num_pairs_filtered'])
    num_geo_pairs = int((num_geos - num_removed_geos) / 2)
    treat_control_removed = (f'{num_geo_pairs}  /  {num_geo_pairs}  /  ' +
                             f'{num_removed_geos}')
    revenue_covered = 100 * baseline / total_response
    proportion_cost_in_experiment = cost_in_experiment / total_spend
    national_budget = human_readable_number(
        budget / proportion_cost_in_experiment)
    designs.append({
        'Budget': human_readable_number(budget),
        'Minimum detectable iROAS': f'{min_detectable_iroas:.3}',
        'Minimum detectable lift in response': f'{min_detectable_lift:.2f} %',
        'Treatment/control/excluded geos': treat_control_removed,
        'Revenue covered by treatment group': f'{revenue_covered:.2f} %',
        'Cost/baseline response': f'{(budget / baseline * 100):.2f} %',
        'Cost if test budget is scaled nationally': national_budget
    })

  designs = pd.DataFrame(designs)
  designs.index.rename('Design', inplace=True)
  return designs


def format_table(
    df: pd.DataFrame,
    formatting_options: List[FormatOptions]) -> style.Styler:
  """Formats a table with the output designs.

  Args:
    df: a table to be formatted.
    formatting_options: a dictionary indicating for each column (key) what
      formatting function to be used and its additional args, e.g.

      formatting_options =
        {'column_1': {'function': fnc, 'args': {'input1': 1, 'input2': 2}}}

  Returns:
    a pandas.io.formats.style.Styler with the table formatted.
  """
  for ind in range(len(formatting_options)):
    tmp_options = formatting_options[ind]
    if ind == 0:
      # if axis is in the args, then the function should be applied on rows/cols
      if 'axis' in tmp_options.args:
        formatted_table = df.style.apply(tmp_options.function,
                                         **tmp_options.args)
      # apply the formatting elementwise
      else:
        formatted_table = df.style.applymap(tmp_options.function,
                                            **tmp_options.args)
    else:
      # if axis is in the args, then the function should be applied on rows/cols
      if 'axis' in tmp_options.args:
        formatted_table = formatted_table.apply(tmp_options.function,
                                                **tmp_options.args)
      # apply the formatting elementwise
      else:
        formatted_table = formatted_table.applymap(tmp_options.function,
                                                   **tmp_options.args)

  return formatted_table


def format_design_table(designs: pd.DataFrame,
                        minimum_detectable_iroas: float,
                        minimum_lift_in_response_metric: float = 10.0,
                        minimum_revenue_covered_by_treatment: float = 5.0):
  """Formats a table with the output designs.

  Args:
    designs: table with columns (Budget, Minimum detectable iROAS,
      Minimum Detectable lift in response, Treatment/control/excluded geos,
      Revenue covered by treatment group, Cost/baseline response,
      Cost if test budget is scaled nationally) containing the output designs,
      e.g. the output of the function create_output_table.
   minimum_detectable_iroas: target minimum detectable iROAS used to define
    the optimality of a design.
   minimum_lift_in_response_metric: threshold minimum detectable lift
    in percentage used to flag designs with higher detectable lift.
   minimum_revenue_covered_by_treatment: value used to flag any design where the
     treatment group is too small based on response.

  Returns:
    a pandas.io.formats.style.Styler with the table formatted.
  """
  formatting_options = [
      FormatOptions(
          column='Minimum detectable lift in response',
          function=flag_percentage_value,
          args={
              'value': minimum_lift_in_response_metric,
              'operation': '>'
          }),
      FormatOptions(
          column='Revenue covered by treatment group',
          function=flag_percentage_value,
          args={
              'value': minimum_revenue_covered_by_treatment,
              'operation': '<'
          }),
      FormatOptions(
          column='Minimum detectable iROAS',
          function=change_background_row,
          args={
              'value': minimum_detectable_iroas,
              'operation': '>',
              'axis': 1
          })
  ]

  return format_table(designs, formatting_options)


def check_input_data(
    data: pd.DataFrame,
    numeric_columns_to_impute: List[str] = None) -> pd.DataFrame:
  """Returns data to be analysed using Trimmed Match with data imputation.

  Args:
    data: data frame with columns (date, geo) and any column specified in
      numeric_columns_to_impute, which should contain at least the columns with
      response and spend information if they have a different name than
      'response' and 'cost', respectively.
    numeric_columns_to_impute: list of columns for which data imputation must be
      performed.

  Returns:
    data frame with columns (date, geo, response, cost) and imputed missing
    data.

  Raises:
    ValueError: if one of the mandatory columns is missing.
  """
  numeric_columns_to_impute = numeric_columns_to_impute or ['response', 'cost']
  mandatory_columns = set(['date', 'geo'] + numeric_columns_to_impute)
  if not mandatory_columns.issubset(data.columns):
    raise ValueError('The mandatory columns ' +
                     f'{mandatory_columns - set(data.columns)} are missing ' +
                     'from the input data.')

  data['date'] = pd.to_datetime(data['date'])
  for column in ['geo'] + numeric_columns_to_impute:
    try:
      data[column] = pd.to_numeric(data[column])
    except:
      raise ValueError(f'Unable to convert column {column} to numeric.')

  geos_and_dates = pd.DataFrame(
      itertools.product(data['date'].unique(), data['geo'].unique()),
      columns=['date', 'geo'])
  data = pd.merge(
      geos_and_dates, data, on=['date', 'geo'],
      how='left').fillna(dict([
          (x, 0) for x in numeric_columns_to_impute
      ])).sort_values(by=['date', 'geo']).reset_index(drop=True)

  return data
