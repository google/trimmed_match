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

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.df = pd.DataFrame({
        'date': ['2020-10-09', '2020-10-10', '2020-10-11'] * 4,
        'geo': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'response': [10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
        'cost': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
    })

  def testCalculateMinDetectableIroasValueError(self):
    with self.assertRaises(ValueError):
      util.CalculateMinDetectableIroas(-0.1, 0.5)
    with self.assertRaises(ValueError):
      util.CalculateMinDetectableIroas(0.1, 1.1)

  def testCalculateMinDetectableIroas(self):
    calc_min_detectable_iroas = util.CalculateMinDetectableIroas(
        significance_level=0.1, power_level=0.9)
    self.assertAlmostEqual(2.56, calc_min_detectable_iroas.at(1.0), places=2)

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

  def testFindFrequencyDataNotSorted(self):
    dates = list(pd.date_range(start='2020-01-01', end='2020-02-01', freq='D'))
    geos = [1, 2, 3, 4]
    df = pd.DataFrame({
        'date': dates * len(geos),
        'geo': sorted(geos * len(dates))
    })
    # permute the order of the rows, so that the dataset is not sorted by date
    df = df.sample(frac=1, replace=False)
    df.set_index(['geo', 'date'], inplace=True)
    frequency = util.infer_frequency(df, 'date', 'geo')
    self.assertEqual(frequency, 'D')

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

  def testFlagPercentageValue(self):
    output = util.flag_percentage_value(val='10 %', value=9.0, operation='>')
    self.assertEqual(output, 'color: red')
    output = util.flag_percentage_value(val='10 %', value=10.1, operation='>')
    self.assertEqual(output, 'color: black')

    output = util.flag_percentage_value(val='10 %', value=9.0, operation='<')
    self.assertEqual(output, 'color: black')
    output = util.flag_percentage_value(val='10 %', value=10.1, operation='<')
    self.assertEqual(output, 'color: red')

  def testCreateOutputTable(self):
    results = pd.DataFrame({
        'num_pairs': [5, 4, 5, 4],
        'experiment_response': [200, 100, 200, 100],
        'experiment_spend': [20, 10, 20, 10],
        'spend_response_ratio': [0.1, 0.1, 0.1, 0.1],
        'budget': [1000, 1000, 500, 500],
        'iroas': [0, 0, 0, 0],
        'rmse': [1, 0.5, 2, 1],
        'rmse_cost_adjusted': [1, 0.625, 2, 1.25],
        'proportion_cost_in_experiment': [1, 0.8, 1, 0.8]
    })
    budgets_for_design = [500, 1000]
    total_response = 300
    total_spend = 25
    geo_treatment = pd.DataFrame({
        'geo': [1, 2, 3, 4],
        'pair': [1, 2, 3, 4],
        'response': [10, 3, 1, 4],
        'spend': [1, 1.5, 0.5, 4]
    })
    average_order_value = 1
    num_geos = 8
    output = util.create_output_table(results, total_response, total_spend,
                                      geo_treatment, budgets_for_design,
                                      average_order_value, num_geos)
    rmse_multiplier = 2.123172
    minimum_detectable_iroas = [rmse_multiplier * 1, rmse_multiplier * 0.5]
    minimum_detectable_lift = [
        minimum_detectable_iroas[x] * budgets_for_design[x] * 100 /
        geo_treatment['response'].sum()
        for x in range(len(minimum_detectable_iroas))
    ]
    minimum_detectable_lift = [f'{x:.2f} %' for x in minimum_detectable_lift]
    minimum_detectable_iroas = [f'{x:.3}' for x in minimum_detectable_iroas]
    expected_output = pd.DataFrame({
        'Budget': ['500', '1K'],
        'Minimum detectable iROAS': minimum_detectable_iroas,
        'Minimum detectable lift in response': minimum_detectable_lift,
        'Treatment/control/excluded geos': ['4  /  4  /  0', '4  /  4  /  0'],
        'Revenue covered by treatment group': ['6.00 %', '6.00 %'],
        'Cost/baseline response': ['2777.78 %', '5555.56 %'],
        'Cost if test budget is scaled nationally': ['1.79K', '3.57K']
    })
    for col in output.columns:
      print(output[col])
      print(expected_output[col])
    self.assertTrue(output.equals(expected_output))

  def testCheckInputData(self):
    temp_df = self.df.copy()
    # remove one observation for geo #2
    temp_df = temp_df[~((temp_df['geo'] == 2) &
                        (temp_df['date'] == '2020-10-10'))]
    geox_data = util.check_input_data(temp_df)
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2020-10-09', '2020-10-10', '2020-10-11'] * 4),
        'geo': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'response': [10, 10, 10, 20, 0.0, 20, 30, 30, 30, 40, 40, 40],
        'cost': [1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
    }).sort_values(by=['date', 'geo']).reset_index(drop=True)
    self.assertTrue(geox_data.equals(expected_df))

  def testCheckInputDataColumns(self):
    temp_df = self.df.copy()
    # remove the column date
    temp_df.drop(columns='date', inplace=True)
    with self.assertRaisesRegex(
        ValueError,
        'The mandatory columns {\'date\'} are missing from the input data'
    ):
      util.check_input_data(temp_df)

  def testCheckInputDataDuplicatedDateGeo(self):
    temp_df = self.df.copy().append(pd.DataFrame(
        {'date': pd.to_datetime(['2020-01-01', '2020-01-01']),
         'geo': [1, 1],
         'response': [0, 1],
         'cost': [0, 1]}))
    with self.assertRaisesRegex(
        ValueError, 'There are duplicated date geo pairs.'
    ):
      util.check_input_data(temp_df)

  def testCheckInputDataUnableToConvertToNumeric(self):
    temp_df = self.df.copy()
    # change the column response to something which cannot be converted to
    # numeric
    temp_df['response'] = ['na', 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40]
    with self.assertRaisesRegex(
        ValueError, 'Unable to convert column response to numeric.'):
      util.check_input_data(temp_df)

  def testCheckInputDataWithMultipleColumnsToImpute(self):
    temp_df = self.df.copy()
    # remove one observation for geo #2
    temp_df = temp_df[~((temp_df['geo'] == 2) &
                        (temp_df['date'] == '2020-10-10'))]
    temp_df['numeric_col'] = 1
    geox_data = util.check_input_data(temp_df,
                                      ['response', 'cost', 'numeric_col'])
    expected_df = pd.DataFrame({
        'date': pd.to_datetime(['2020-10-09', '2020-10-10', '2020-10-11'] * 4),
        'geo': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'response': [10, 10, 10, 20, 0.0, 20, 30, 30, 30, 40, 40, 40],
        'cost': [1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        'numeric_col': [
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0
        ],
    }).sort_values(by=['date', 'geo']).reset_index(drop=True)
    self.assertTrue(geox_data.equals(expected_df))

  def testPairingNotInAList(self):
    """Checks an error is raised if pairs are not passed in a list."""
    # geo 1 and 2 appear in two pairs.
    pairs = pd.DataFrame({
        'geo1': [1, 2, 2],
        'geo2': [3, 4, 1],
        'pair': [1, 2, 3]
    })
    with self.assertRaisesRegex(ValueError,
                                r'pairs must be a list of dataframes.'):
      util.check_pairs(
          pretest_data=self.df,
          pairs=pairs)

  def testPairingWithDuplicatedGeos(self):
    """Checks an error is raised if a geo appears in multiple pairs."""
    # geo 1 and 2 appear in two pairs.
    pairs = [
        pd.DataFrame({
            'geo1': [1, 2, 2],
            'geo2': [3, 4, 1],
            'pair': [1, 2, 3]
        })
    ]
    with self.assertRaisesRegex(
        ValueError, f'Some geos are duplicated in the pairing {pairs[0]}.'):
      util.check_pairs(
          pretest_data=self.df,
          pairs=pairs)

  def testPairingWithMoreThanTwoGeosPerPair(self):
    """Checks an error is raised if a pair appears multiple times."""
    # geo 1 and 2 appear in two pairs.
    pairs = [
        pd.DataFrame({
            'geo1': [1, 2],
            'geo2': [3, 4],
            'pair': [1, 1]
        })
    ]
    with self.assertRaisesRegex(
        ValueError, r'a pair should only have two geos.'):
      util.check_pairs(
          pretest_data=self.df,
          pairs=pairs)

  def testPairingWithGeosNotInPretestData(self):
    """Raises an error if a geo appears in the pairs but not in the data."""
    # geo 5 and 6 appear in the pairs but not in the pretest data.
    pairs = [pd.DataFrame({
        'geo1': [1, 2, 5],
        'geo2': [3, 4, 6],
        'pair': [1, 2, 3]
    })]
    with self.assertRaisesRegex(ValueError,
                                r'The geos ' +
                                r'{5, 6} appear ' +
                                r'in the pairs but not in the pretest data.'):
      util.check_pairs(
          pretest_data=self.df,
          pairs=pairs)

if __name__ == '__main__':
  unittest.main()
