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

import datetime
import numpy as np
import pandas as pd

from trimmed_match.design import util
import unittest


class UtilTest(unittest.TestCase):

  def testFindDaysToExclude(self):
    day_week_exclude = [
        '202010', '202011', '202013', '20200810', '20200814'
    ]
    days_to_remove = util.find_days_to_exclude(day_week_exclude)
    temp_days = ['2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05',
                 '2020-03-06', '2020-03-07', '2020-03-08', '2020-03-09',
                 '2020-03-10', '2020-03-11', '2020-03-12', '2020-03-13',
                 '2020-03-14', '2020-03-15', '2020-03-23', '2020-03-24',
                 '2020-03-25', '2020-03-26', '2020-03-27', '2020-03-28',
                 '2020-03-29', '2020-08-10', '2020-08-14']
    expected_days = [
        datetime.datetime.strptime(x, '%Y-%m-%d') for x in temp_days
    ]
    self.assertCountEqual(days_to_remove, expected_days)

  def testNoDuplicateDays(self):
    day_week_exclude = ['202010', '20200303']
    days_to_remove = util.find_days_to_exclude(day_week_exclude)
    temp_days = ['2020-03-02', '2020-03-03', '2020-03-04', '2020-03-05',
                 '2020-03-06', '2020-03-07', '2020-03-08']
    expected_days = [
        datetime.datetime.strptime(x, '%Y-%m-%d') for x in temp_days
    ]
    self.assertCountEqual(days_to_remove, expected_days)

  def testWrongDateFormat(self):
    day_week_exclude = ['202010', '2020-03-03']
    with self.assertRaises(ValueError):
      util.find_days_to_exclude(day_week_exclude)

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

if __name__ == '__main__':
  unittest.main()
