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

"""Tests for trimmed_match.design.common_classes."""

import pandas as pd

from trimmed_match.design import common_classes
import unittest

GeoXType = common_classes.GeoXType
GeoLevelData = common_classes.GeoLevelData
GeoLevelPotentialOutcomes = common_classes.GeoLevelPotentialOutcomes
TimeWindow = common_classes.TimeWindow
FormatOptions = common_classes.FormatOptions


class CommonClassesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._geox_outcome = GeoLevelData(1, 1.0, 2.0)
    self._potential_outcomes = GeoLevelPotentialOutcomes(
        GeoLevelData(1, 1.0, 2.0), GeoLevelData(1, 3.0, 4.0))
    self._t1 = '2019-01-01'
    self._t2 = '2020-01-01'

  def testGeoXType(self):
    for x in [
        'CONTROL', 'GO_DARK', 'HEAVY_UP', 'HEAVY_DOWN', 'HOLD_BACK',
        'GO_DARK_TREATMENT_NOT_BAU_CONTROL'
    ]:
      self.assertIn(x, GeoXType.__members__)
    self.assertNotIn('go-dark', GeoXType.__members__)
    self.assertEqual(GeoXType['GO_DARK'], GeoXType.GO_DARK)

  def testGeoLevelData(self):
    self.assertEqual(1.0, self._geox_outcome.response)
    self.assertEqual(2.0, self._geox_outcome.spend)

  def testGeoLevelPotentialOutcomes(self):
    self.assertEqual(1.0, self._potential_outcomes.controlled.response)
    self.assertEqual(2.0, self._potential_outcomes.controlled.spend)
    self.assertEqual(3.0, self._potential_outcomes.treated.response)
    self.assertEqual(4.0, self._potential_outcomes.treated.spend)

  def testTimeWindow(self):
    result1 = TimeWindow(self._t1, self._t2)
    dates = pd.to_datetime(
        pd.Series([
            '2019-09-01', '2019-10-01', '2019-11-01', '2019-12-01',
            '2020-01-01', '2020-01-01', '2020-02-01', '2020-03-01'
        ]))
    is_contained = pd.Series([True, True, True, True, True, True, False, False])
    latest_dates = pd.to_datetime(
        pd.Series([
            '2020-03-01', '2020-02-01', '2020-01-01', '2019-12-01', '2019-11-01'
        ]))
    self.assertEqual(pd.Timestamp(self._t1), result1.first_day)
    self.assertEqual(pd.Timestamp(self._t2), result1.last_day)
    self.assertTrue(result1.contains(dates).equals(is_contained))
    self.assertTrue(
        result1.finds_latest_date_moving_window_in(dates).reset_index(
            drop=True).equals(latest_dates))
    with self.assertRaises(ValueError):
      result1.finds_latest_date_moving_window_in(
          pd.to_datetime(pd.Series(['2020-02-01', '2020-03-01'])))
    self.assertTrue(
        result1.is_included_in(
            TimeWindow(pd.Timestamp('2018-10-01'), pd.Timestamp('2021-12-01'))))
    self.assertFalse(
        result1.is_included_in(
            TimeWindow(pd.Timestamp('2019-10-01'), pd.Timestamp('2021-12-01'))))

  def testFormatOptions(self):
    # checks the correct initialization of the class.
    options = FormatOptions(
        column='Minimum detectable lift in response',
        function=pd.to_numeric,
        args={
            'value': 3.0,
            'operation': '>'
        })
    self.assertEqual(options.column, 'Minimum detectable lift in response')
    self.assertEqual(options.function, pd.to_numeric)
    self.assertDictEqual(
        options.args, {
            'value': 3.0,
            'operation': '>',
            'subset': 'Minimum detectable lift in response'
        })


class TimeWindowShiftedTest(unittest.TestCase):

  def testTimeWindowShiftedFrom(self):
    tw = TimeWindow('2020-01-01', '2020-01-10')
    new_first_day = pd.Timestamp('2021-01-01')
    result = common_classes.time_window_shifted_from(tw, new_first_day)
    expected = TimeWindow('2021-01-01', '2021-01-10')
    self.assertEqual(expected.first_day, result.first_day)
    self.assertEqual(expected.last_day, result.last_day)


if __name__ == '__main__':
  unittest.main()
