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

"""Tests for trimmed_match.design.tests.trimmed_match_design."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trimmed_match.design import common_classes
from trimmed_match.design import matched_pairs_rmse
from trimmed_match.design import trimmed_match_design
import unittest

GeoXType = common_classes.GeoXType
TimeWindow = common_classes.TimeWindow
TrimmedMatchGeoXDesign = trimmed_match_design.TrimmedMatchGeoXDesign
MatchedPairsRMSE = matched_pairs_rmse.MatchedPairsRMSE


class TrimmedMatchDesignTest(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super(TrimmedMatchDesignTest, self).setUp()
    self.test_data = pd.DataFrame({
        'date':
            pd.to_datetime([
                '2019-01-01', '2019-10-01', '2019-01-01', '2019-10-01',
                '2019-01-01', '2019-10-01', '2019-01-01', '2019-10-01'
            ]),
        'geo': [1, 1, 2, 2, 3, 3, 4, 4],
        'response': [1, 2, 2, 5, 1, 2, 3, 4],
        'spend': [1, 1.5, 2, 2.5, 1, 1.5, 5, 6],
        'metric': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    })

    geos = list(range(5, 23)) * 2
    geos.sort()

    self.add_pair = pd.DataFrame({
        'date':
            pd.to_datetime(
                ['2019-01-01', '2019-10-01'] * 18),
        'geo': geos,
        'response': [10, 20] * 18,
        'spend': [10, 10] * 18
    })

    self.design_window = TimeWindow(
        pd.Timestamp('2019-01-01'), pd.Timestamp('2019-10-01'))
    self.evaluation_window = TimeWindow(
        pd.Timestamp('2019-09-01'), pd.Timestamp('2019-10-01'))
    self.test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0})

  def testMatchingMetrics(self):
    self.assertDictEqual(self.test_class._matching_metrics, {
        'response': 1.0,
        'spend': 0.0,
    })
    default_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window)
    self.assertDictEqual(default_class._matching_metrics, {
        'response': 1.0,
        'spend': 0.01,
    })

  def testPropertyGetter(self):
    # test that they are None at initialization of the class
    self.assertIsNone(self.test_class.pairs)
    self.assertIsNone(self.test_class.geo_level_eval_data)
    # test the values are updated once the pairs have been created
    pairs, geo_level_eval_data = self.test_class.create_geo_pairs(
        use_cross_validation=False)

    self.assertTrue(pairs.equals(self.test_class.pairs))
    self.assertTrue(
        geo_level_eval_data.equals(self.test_class.geo_level_eval_data))

  def testMissingResponseVariable(self):
    with self.assertRaises(ValueError):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          response='revenue',
          matching_metrics={'response': 1.0})

  def testMissingSpendProxy(self):
    with self.assertRaises(ValueError):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          spend_proxy='cost',
          matching_metrics={'response': 1.0})

  def testInvalidResponseVariable(self):
    with self.assertRaises(ValueError):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          response='unknown',
          matching_metrics={'response': 1.0})

  def testInvalidSpendProxy(self):
    with self.assertRaises(ValueError):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          spend_proxy='unknown',
          matching_metrics={'response': 1.0})

  def testZeroSpendProxy(self):
    data = self.test_data.copy()
    data['spend'] = 0
    with self.assertRaisesRegex(
        ValueError,
        r'The column spend should have some positive entries. '
        'The sum of spend found is 0.0'):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          spend_proxy='spend',
          matching_metrics={'response': 1.0})

  def testInvalidValues(self):
    pretest_data = self.test_data.copy()
    pretest_data['revenue'] = [1, 2, 3, 4, 5, 6, 7, 'nan']
    with self.assertRaises(ValueError):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=pretest_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          response='revenue',
          matching_metrics={'response': 1.0})

  def testOddNumberOfGeos(self):
    add_geo = pd.DataFrame({
        'date':
            pd.to_datetime(
                ['2019-01-01', '2019-10-01']),
        'geo': [5, 5],
        'response': [1, 2],
        'spend': [1, 1.5]
    })

    pretest_data = pd.concat(
        [self.test_class._pretest_data, add_geo], sort=False)
    with self.assertRaises(ValueError):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=pretest_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          matching_metrics={'response': 1.0})

  def testCreateSignTestData(self):
    expected = pd.DataFrame({
        'geo': [1, 2, 3, 4],
        'response': [2, 5, 2, 4],
        'spend': [1.5, 2.5, 1.5, 6.0]
    })
    result = self.test_class._create_sign_test_data()
    self.assertTrue(result.equals(expected))

  def testCreateGeoPairsNoCV(self):
    pairs, geo_level_eval_data = self.test_class.create_geo_pairs(
        use_cross_validation=False)
    geo_level_eval_data.sort_values(by='geo', inplace=True)
    geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [1, 2, 1, 2],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        pairs.equals(
            pd.DataFrame({
                'geo1': [1, 2],
                'geo2': [3, 4],
                'distance': [0.0, 0.0],
                'pair': [1, 2]
            })))

  def testCreateGeoPairsCV(self):
    pairs, geo_level_eval_data = self.test_class.create_geo_pairs(
        use_cross_validation=True)
    geo_level_eval_data.sort_values(by='geo', inplace=True)
    geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [2, 1, 2, 1],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        pairs.equals(
            pd.DataFrame({
                'geo1': [4, 1],
                'geo2': [2, 3],
                'distance': [1/7, 0.0],
                'pair': [1, 2]
            })))

  def testCheckPairsHaveCorrectOrder(self):
    add_geo = pd.DataFrame({
        'date':
            pd.to_datetime(
                ['2019-01-01', '2019-10-01', '2019-01-01', '2019-10-01']),
        'geo': [5, 5, 6, 6],
        'response': [4.45, 20, 4.55, 20],
        'spend': [10, 10, 10, 10]
    })

    self.test_class._pretest_data = pd.concat(
        [self.test_class._pretest_data, add_geo], sort=False)
    pairs, geo_level_eval_data = self.test_class.create_geo_pairs(
        use_cross_validation=True)
    geo_level_eval_data.sort_values(by='geo', inplace=True)
    geo_level_eval_data.reset_index(drop=True, inplace=True)
    pairs = pairs.round({'distance': 5})
    self.assertTrue(
        geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4, 5, 6],
                'pair': [3, 1, 3, 1, 2, 2],
                'response': [2.0, 5.0, 2.0, 4.0, 20.0, 20.0],
                'spend': [1.5, 2.5, 1.5, 6, 10, 10]
            })))
    self.assertTrue(
        pairs.equals(
            pd.DataFrame({
                'geo1': [4, 6, 1],
                'geo2': [2, 5, 3],
                'distance': [1/16, 0.1/16, 0.0],
                'pair': [1, 2, 3]
            })))

  def testCreateGeoPairsMatchingMetric(self):
    self.test_class._matching_metrics = {'response': 1.0, 'spend': 0.5}
    pairs, geo_level_eval_data = self.test_class.create_geo_pairs(
        use_cross_validation=True)
    geo_level_eval_data.sort_values(by='geo', inplace=True)
    geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [2, 1, 2, 1],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        pairs.equals(
            pd.DataFrame({
                'geo1': [4, 1],
                'geo2': [2, 3],
                'distance': [1/7+1/6, 0.0],
                'pair': [1, 2]
            })))

  def testCreateGeoPairsMatchingMetricUserDistance(self):
    self.test_class._matching_metrics = {
        'metric': 1.0,
        'response': 0,
        'spend': 0
    }
    pairs, geo_level_eval_data = self.test_class.create_geo_pairs(
        use_cross_validation=True)
    geo_level_eval_data.sort_values(by='geo', inplace=True)
    geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [1, 1, 2, 2],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        pairs.equals(
            pd.DataFrame({
                'geo1': [2, 4],
                'geo2': [1, 3],
                'distance': [1/8, 1/8],
                'pair': [1, 2]
            })))

  def testReportCandidateDesign(self):
    """Checks the calculation with zero RMSE."""

    np.random.seed(0)
    self.test_class._pretest_data = pd.concat(
        [self.test_class._pretest_data, self.add_pair], sort=False)
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      self.test_class._geox_type = geox_type
      (results, detailed_results) = self.test_class.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=1000)

      expected_results = pd.DataFrame({
          'num_pairs_filtered': [0, 1] * 4,
          'experiment_response': [373, 364] * 4,
          'experiment_spend': [191.5, 183] * 4,
          'spend_response_ratio': [191.5 / 373, 183 / 364] * 4,
          'budget': [30, 30, 40, 40, 30, 30, 40, 40],
          'iroas': [0, 0, 0, 0, 2, 2, 2, 2],
          'proportion_cost_in_experiment': [1, 183 / 191.5] * 4
      })

      self.assertTrue(results[[
          'num_pairs_filtered', 'experiment_response', 'experiment_spend',
          'spend_response_ratio', 'budget', 'iroas',
          'proportion_cost_in_experiment'
      ]].equals(expected_results))
      for (_, iroas, _), value in detailed_results.items():
        self.assertAlmostEqual(
            value['estimate'].mean(), iroas, delta=0.05 + 0.1 * iroas)

  def testInsufficientNumberOfGeos(self):
    with self.assertRaises(ValueError) as cm:
      self.test_class.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=1000)
    self.assertEqual(
        str(cm.exception),
        'the number of geos to design an experiment must be >= 20, got 4')

  def testPlotCandidateDesign(self):
    """Check the function plot_candidate_design outputs a dict of axes."""

    np.random.seed(0)
    self.test_class._pretest_data = pd.concat(
        [self.test_class._pretest_data, self.add_pair], sort=False)
    self.test_class._geox_type = GeoXType.HEAVY_UP
    budget_list = [30, 40]
    iroas_list = [0, 2]
    results, _ = self.test_class.report_candidate_designs(
        budget_list=budget_list,
        iroas_list=iroas_list,
        use_cross_validation=True,
        num_pairs_filtered_list=[0, 1],
        num_simulations=100)

    axes_dict = self.test_class.plot_candidate_design(results)
    for budget in budget_list:
      for iroas in iroas_list:
        self.assertIsInstance(axes_dict[(budget, iroas)], plt.Figure)

  def testOutputCandidateDesign(self):
    """Check that the design in output is ok when group ids are specified."""
    self.test_class._pretest_data = pd.DataFrame({
        'date':
            pd.to_datetime(
                ['2019-01-01', '2019-10-01'] * 20),
        'geo': sorted(list(range(20)) * 2),
        'response': range(100, 140),
        'spend': range(40)
    })

    _ = self.test_class.report_candidate_designs(
        budget_list=[30],
        iroas_list=[0],
        use_cross_validation=True,
        num_pairs_filtered_list=[0],
        num_simulations=200)
    _ = self.test_class.output_chosen_design(num_pairs_filtered=0, base_seed=0)
    default_ids = self.test_class.geo_level_eval_data
    _ = self.test_class.output_chosen_design(
        num_pairs_filtered=0, base_seed=0, group_control=2, group_treatment=1)
    specific_ids = self.test_class.geo_level_eval_data

    self.assertTrue(
        default_ids.drop('assignment', axis=1).equals(
            specific_ids.drop('assignment', axis=1)))
    self.assertTrue(
        np.array_equal(default_ids['assignment'].values,
                       2 - specific_ids['assignment'].values))


if __name__ == '__main__':
  unittest.main()
