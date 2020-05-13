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

from typing import List
import numpy as np
import pandas as pd


def find_days_to_exclude(
    day_week_exclude: List[str]) -> List['datetime.datetime']:
  """Return a list of dates to exclude from a list of days and weeks.

  Args:
    day_week_exclude: a List of strings with format indicating a single day as
    '20200101' (YYYYMMDD) or an entire week as '202010' (YYYYWW). A week starts
    on a Monday and ends on a Sunday.

  Returns:
    days_exclude: a List of datetime.datetime days to obtained by expanding the
      list in input.

  """
  days_exclude = []
  for x in day_week_exclude:
    if len(x) == 8:
      days_exclude += [datetime.datetime.strptime(x, '%Y%m%d')]
    elif len(x) == 6:
      start = datetime.datetime.strptime(x[:4] + '-W' + x[4:] + '-1',
                                         '%G-W%V-%u')
      for i in range(7):
        days_exclude += [start + datetime.timedelta(days=i)]
    else:
      raise ValueError(f"""The elements of the list should have format "YYYYWW"
                       or "YYYYMMDD", got {x}""")
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
