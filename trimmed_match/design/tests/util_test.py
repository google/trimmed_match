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

"""Tests for trimmed_match.design.tests.util."""

import numpy as np
import pandas as pd

from trimmed_match.design import common_classes
from trimmed_match.design import util
import unittest

TimeWindow = common_classes.TimeWindow


class UtilTest(unittest.TestCase):

  def testFindDaysToExclude(self):
    day_week_exclude = [
        '2020/10/10', '2020/11/10-2020/12/10', '2020/08/10']
    days_to_remove = util.find_days_to_exclude(day_week_exclude)
    expected_days = [
        TimeWindow(pd.Timestamp('2020-10-10'), pd.Timestamp('2020-10-10')),
        TimeWindow(pd.Timestamp('2020-11-10'), pd.Timestamp('2020-12-10')),
        TimeWindow(pd.Timestamp('2020-08-10'), pd.Timestamp('2020-08-10')),
    ]
    for x in range(len(expected_days)):
      self.assertEqual(days_to_remove[x].first_day, expected_days[x].first_day)
      self.assertEqual(days_to_remove[x].last_day, expected_days[x].last_day)

  def testWrongDateFormat(self):
    incorrect_day = ['2020/13/13', '2020/03/03']
    with self.assertRaises(ValueError):
      util.find_days_to_exclude(incorrect_day)

    incorrect_time_window = ['2020/10/13 - 2020/13/11', '2020/03/03']
    with self.assertRaises(ValueError):
      util.find_days_to_exclude(incorrect_time_window)

    incorrect_format = ['2020/10/13 - 2020/13/11 . 2020/10/10']
    with self.assertRaises(ValueError):
      util.find_days_to_exclude(incorrect_format)

  def testExpandTimeWindows(self):
    day_week_exclude = [
        '2020/10/10', '2020/11/10-2020/12/10', '2020/08/10']
    days_to_remove = util.find_days_to_exclude(day_week_exclude)
    periods = util.expand_time_windows(days_to_remove)
    expected = [
        pd.Timestamp('2020-10-10', freq='D'),
        pd.Timestamp('2020-08-10', freq='D'),
    ]
    expected += pd.date_range(start='2020-11-10', end='2020-12-10', freq='D')
    self.assertEqual(len(periods), len(expected))
    for x in periods:
      self.assertIn(x, expected)

  def testCheckNoOverlap(self):
    expected = 0.0
    dates_left = ['2020-03-02', '2020-03-03']
    dates_right = ['2020-03-04', '2020-03-05']
    percentage = util.overlap_percent(dates_left, dates_right)
    self.assertEqual(percentage, expected)

  def testCheckOverlap(self):
    expected = 50.0
    dates_left = ['2020-03-02', '2020-03-04']
    dates_right = ['2020-03-04', '2020-03-05']
    percentage = util.overlap_percent(dates_left, dates_right)
    self.assertEqual(percentage, expected)

  def testFindFrequency(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='D'))
    geos = [1, 2, 3, 4]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    df.set_index(['geo', 'date'], inplace=True)
    frequency = util.infer_frequency(df, 'date', 'geo')
    self.assertEqual(frequency, 'D')

    weeks = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='W'))
    df = pd.DataFrame({
        'date': weeks * len(geos),
        'geo': sorted(geos * len(weeks))
    })
    df.set_index(['geo', 'date'], inplace=True)
    frequency = util.infer_frequency(df, 'date', 'geo')
    self.assertEqual(frequency, 'W')

  def testDifferentFrequencies(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='D'))
    weeks = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='W'))
    geos = [1] * len(dates) + [2] * len(weeks)
    df = pd.DataFrame({
        'date': dates + weeks,
        'geo': geos
    })
    df.set_index(['geo', 'date'], inplace=True)
    with self.assertRaises(ValueError) as cm:
      _ = util.infer_frequency(df, 'date', 'geo')
    self.assertEqual(
        str(cm.exception),
        'The provided time series seem to have irregular frequencies.')

  def testInsufficientData(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-01-01', freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    df.set_index(['geo', 'date'], inplace=True)
    with self.assertRaises(ValueError) as cm:
      _ = util.infer_frequency(df, 'date', 'geo')
    self.assertEqual(
        str(cm.exception),
        'At least one series with more than one observation must be provided.')

  def testUnknownFrequency(self):
    dates = list(pd.to_datetime(['2020-10-10', '2020-10-13', '2020-10-16']))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    df.set_index(['geo', 'date'], inplace=True)
    with self.assertRaises(ValueError) as cm:
      _ = util.infer_frequency(df, 'date', 'geo')
    self.assertEqual(str(cm.exception),
                     'Frequency could not be identified. Got 3 days.')

  def testNoMissingDates(self):
    dates = list(pd.date_range(start='2020-01-01', periods=28, freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    missing = util.find_missing_dates(
        geox_data=df,
        start_date=pd.Timestamp('2020-01-01'),
        period_duration_weeks=4,
        number_of_observations=28,
        frequency='D')
    self.assertListEqual(missing, [])

  def testFindMissingDates(self):
    dates = list(pd.date_range(start='2020-01-01', periods=28, freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    missing = util.find_missing_dates(
        geox_data=df,
        start_date=pd.Timestamp('2020-01-02'),
        period_duration_weeks=4,
        number_of_observations=28,
        frequency='D')
    self.assertEqual(missing, np.array(['2020-01-29']))

  def testCheckValidPeriods(self):
    dates = list(pd.date_range(start='2020-01-01', periods=28*2, freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    out = util.check_time_periods(
        geox_data=df,
        start_date_eval=pd.Timestamp('2020-01-01'),
        start_date_aa_test=pd.Timestamp('2020-01-29'),
        experiment_duration_weeks=4,
        frequency='D')
    self.assertTrue(out)

  def testCheckValidPeriodsInferredFrequency(self):
    dates = list(pd.date_range(start='2020-01-01', periods=28*2, freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    out = util.check_time_periods(
        geox_data=df,
        start_date_eval=pd.Timestamp('2020-01-01'),
        start_date_aa_test=pd.Timestamp('2020-01-29'),
        experiment_duration_weeks=4,
        frequency='infer')
    self.assertTrue(out)

  def testCheckValidPeriodsWeekly(self):
    dates = list(pd.date_range(start='2020-01-01', periods=4*2, freq='W'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    out = util.check_time_periods(
        geox_data=df,
        start_date_eval=pd.Timestamp('2020-01-01'),
        start_date_aa_test=pd.Timestamp('2020-01-29'),
        experiment_duration_weeks=4,
        frequency='W')
    self.assertTrue(out)
    out_infer = util.check_time_periods(
        geox_data=df,
        start_date_eval=pd.Timestamp('2020-01-01'),
        start_date_aa_test=pd.Timestamp('2020-01-29'),
        experiment_duration_weeks=4,
        frequency='infer')
    self.assertTrue(out_infer)

  def testInvalidPeriods(self):
    dates = list(pd.date_range(start='2020-01-01', periods=27 * 2, freq='D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    missing = ['2020-02-24', '2020-02-25']
    freq_str = 'days'
    with self.assertRaises(ValueError) as cm:
      _ = util.check_time_periods(
          geox_data=df,
          start_date_eval=pd.Timestamp('2020-01-01'),
          start_date_aa_test=pd.Timestamp('2020-01-29'),
          experiment_duration_weeks=4,
          frequency='D')
    self.assertEqual(
        str(cm.exception),
        (f'The AA test period contains the following {freq_str} ' +
         f'{missing} for which we do not have data.'))

  def testInvalidPeriodsWeekly(self):
    dates = list(pd.date_range(start='2020-01-01', periods=7, freq='7D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    missing = ['2020-02-19']
    freq_str = 'weeks'
    with self.assertRaises(ValueError) as cm:
      _ = util.check_time_periods(
          geox_data=df,
          start_date_eval=pd.Timestamp('2020-01-01'),
          start_date_aa_test=pd.Timestamp('2020-01-29'),
          experiment_duration_weeks=4,
          frequency='W')
    self.assertEqual(
        str(cm.exception),
        (f'The AA test period contains the following {freq_str} ' +
         f'{missing} for which we do not have data.'))

  def testInvalidPeriodsWeeklyMiddle(self):
    dates = list(pd.date_range(start='2020-01-01', periods=8, freq='7D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    df.drop(df[df['date'] == '2020-01-08'].index, inplace=True)
    missing = ['2020-01-08']
    freq_str = 'weeks'
    with self.assertRaises(ValueError) as cm:
      _ = util.check_time_periods(
          geox_data=df,
          start_date_eval=pd.Timestamp('2020-01-01'),
          start_date_aa_test=pd.Timestamp('2020-01-29'),
          experiment_duration_weeks=4,
          frequency='W')
    self.assertEqual(
        str(cm.exception),
        (f'The evaluation period contains the following {freq_str} ' +
         f'{missing} for which we do not have data.'))

  def testInvalidFrequency(self):
    dates = list(pd.date_range(start='2020-01-01', periods=8, freq='7D'))
    geos = [1, 2]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    with self.assertRaises(ValueError) as cm:
      _ = util.check_time_periods(
          geox_data=df,
          start_date_eval=pd.Timestamp('2020-01-01'),
          start_date_aa_test=pd.Timestamp('2020-01-29'),
          experiment_duration_weeks=4,
          frequency='invalid')
    self.assertEqual(
        str(cm.exception),
        'frequency should be one of ["infer", "D", "W"], got invalid')

  def testHumanReadableFormat(self):
    numbers = [123, 10765, 13987482, 8927462746, 1020000000000]
    numb_formatted = [
        util.human_readable_number(num) for num in numbers
    ]
    self.assertEqual(numb_formatted, ['123', '10.8K', '14M', '8.93B', '1.02tn'])


if __name__ == '__main__':
  unittest.main()
