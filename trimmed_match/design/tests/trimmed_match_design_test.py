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

    self.nontrivial_data = pd.DataFrame({
        'date':
            pd.to_datetime(
                ['2019-01-01', '2019-10-01'] * 20),
        'geo': sorted(list(range(1, 21)) * 2),
        'response': range(100, 140),
        'spend': range(1, 41)
    })

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

  def testMatchingMetricsZeroWeights(self):
    """Checks that an error is raised if all weights are 0 in matching metrics."""
    with self.assertRaisesRegex(
        ValueError,
        r'Weights in matching_metrics sum up to 0.'):
      TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          matching_metrics={
              'response': 0.0,
              'cost': 0.0
          })

  def testMatchingMetricsWithCustomColumnNames(self):
    default_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'metric': 1.0})

    self.assertDictEqual(default_class._matching_metrics, {
        'response': 0.0,
        'spend': 0.0,
        'metric': 1.0,
    })

  def testPropertyGetter(self):
    # test that they are None at initialization of the class
    self.assertIsNone(self.test_class.pairs)
    self.assertIsNone(self.test_class.geo_level_eval_data)
    # test the values are updated once the pairs have been created
    self.test_class.create_geo_pairs(use_cross_validation=False)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        self.test_class.pairs.equals(
            pd.DataFrame({
                'geo1': [1, 2],
                'geo2': [3, 4],
                'distance': [0.0, 0.0],
                'pair': [1, 2]
            })))
    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [1, 2, 1, 2],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))

  def testMissingResponseVariable(self):
    with self.assertRaisesRegex(
        ValueError,
        r'revenue is not available in pretest_data'):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          response='revenue',
          matching_metrics={'response': 1.0})

  def testMissingSpendProxy(self):
    with self.assertRaisesRegex(
        ValueError,
        r'cost is not available in pretest_data'):
      _ = TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          spend_proxy='cost',
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
    with self.assertRaisesRegex(
        ValueError,
        r'Unable to convert column revenue to numeric.'):
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
    with self.assertRaisesRegex(
        ValueError,
        r'Number of geos must be even, but got 5.'):
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
    self.test_class.create_geo_pairs(use_cross_validation=False)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [1, 2, 1, 2],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        self.test_class.pairs.equals(
            pd.DataFrame({
                'geo1': [1, 2],
                'geo2': [3, 4],
                'distance': [0.0, 0.0],
                'pair': [1, 2]
            })))

  def testCreateGeoPairsCV(self):
    self.test_class.create_geo_pairs(use_cross_validation=True)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [2, 1, 2, 1],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        self.test_class.pairs.equals(
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
    self.test_class.create_geo_pairs(use_cross_validation=True)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    pairs = self.test_class.pairs.round({'distance': 5})
    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
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
    self.test_class.create_geo_pairs(use_cross_validation=True)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [2, 1, 2, 1],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        self.test_class.pairs.equals(
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
    self.test_class.create_geo_pairs(use_cross_validation=True)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [1, 1, 2, 2],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6]
            })))
    self.assertTrue(
        self.test_class.pairs.equals(
            pd.DataFrame({
                'geo1': [2, 4],
                'geo2': [1, 3],
                'distance': [1/8, 1/8],
                'pair': [1, 2]
            })))

  def testPassPairingWithDuplicatedGeos(self):
    # geo 1 and 2 appear in two pairs.
    pairs = pd.DataFrame({
        'geo1': [1, 2, 2],
        'geo2': [3, 4, 1],
        'pair': [1, 2, 3]
    })
    with self.assertRaisesRegex(ValueError,
                                r'Some geos are duplicated in the pairs.'):
      TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          matching_metrics={'response': 1.0},
          pairs=pairs)

  def testPassPairingWithGeosNotInPretestData(self):
    # geo 5 and 6 appear in the pairs but not in the pretest data.
    pairs = pd.DataFrame({
        'geo1': [1, 2, 5],
        'geo2': [3, 4, 6],
        'pair': [1, 2, 3]
    })
    with self.assertRaisesRegex(ValueError,
                                r'The geos ' +
                                r'{5, 6} appear ' +
                                r'in the pairs but not in the pretest data.'):
      TrimmedMatchGeoXDesign(
          GeoXType.HEAVY_UP,
          pretest_data=self.test_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          matching_metrics={'response': 1.0},
          pairs=pairs)

  def testCreateGeoLevelDataWhenPairsArePassed(self):
    pairs = pd.DataFrame({
        'geo1': [1, 2],
        'geo2': [3, 4],
        'pair': [1, 2]
    })
    test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0},
        pairs=pairs)
    test_class.create_geo_level_eval_data()
    test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)
    self.assertTrue(
        test_class.geo_level_eval_data.sort_index(axis=1).equals(
            pd.DataFrame({
                'geo': [1, 2, 3, 4],
                'pair': [1, 2, 1, 2],
                'response': [2, 5, 2, 4],
                'spend': [1.5, 2.5, 1.5, 6.0]
            })))
    self.assertTrue(
        test_class.pairs.equals(
            pd.DataFrame({
                'geo1': [1, 2],
                'geo2': [3, 4],
                'pair': [1, 2]
            })))

  def testCreateGeoLevelDataRaisesError(self):
    test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0}
        )
    with self.assertRaisesRegex(ValueError, r'pairs are not specified'):
      test_class.create_geo_level_eval_data()

  def testCheckSameGeoLevelDataWhenPairsAreSpecifiedOrNot(self):
    self.test_class._matching_metrics = {'response': 1.0, 'spend': 0.5}
    self.test_class.create_geo_pairs(use_cross_validation=True)
    self.test_class.create_geo_level_eval_data()
    self.test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    self.test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)

    pairs_specified = pd.DataFrame({
        'geo1': [1, 4],
        'geo2': [3, 2],
        'pair': [2, 1]
    })
    test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0, 'spend': 0.5},
        pairs=pairs_specified)
    test_class.create_geo_level_eval_data()
    test_class.geo_level_eval_data.sort_values(by='geo', inplace=True)
    test_class.geo_level_eval_data.reset_index(drop=True, inplace=True)

    self.assertTrue(
        self.test_class.geo_level_eval_data.sort_index(axis=1).equals(
            test_class.geo_level_eval_data.sort_index(axis=1)))
    self.assertTrue(self.test_class.pairs[['geo1', 'geo2', 'pair']].equals(
        test_class.pairs.reset_index(drop=True)))

  def testSameRMSEWhenPairsAreSpecifiedOrNot(self):

    test_class1 = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.nontrivial_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0},
        pairs=None)

    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue

      test_class1._geox_type = geox_type
      _, expected_detailed_results = test_class1.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=100)

      test_class2 = TrimmedMatchGeoXDesign(
          geox_type,
          pretest_data=self.nontrivial_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          matching_metrics={'response': 1.0},
          pairs=test_class1.pairs)
      _, detailed_results = test_class2.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=100)

      for key in detailed_results.keys():
        self.assertTrue(
            np.array_equal(detailed_results[key]['estimate'],
                           expected_detailed_results[key]['estimate']))

  def testSameRMSEWhenPairsAreSpecifiedOrNotDifferentGeoOrder(self):
    test_class1 = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.nontrivial_data,
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0},
        pairs=None)

    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue

      test_class1._geox_type = geox_type
      _, expected_detailed_results = test_class1.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=100)

      # change the order of geo1 and geo2 in some of the pairs
      pairs = test_class1.pairs.copy()
      pairs.loc[0:3, 'geo1'] = test_class1.pairs.loc[0:3, 'geo2']
      pairs.loc[0:3, 'geo2'] = test_class1.pairs.loc[0:3, 'geo1']

      test_class2 = TrimmedMatchGeoXDesign(
          geox_type,
          pretest_data=self.nontrivial_data,
          time_window_for_design=self.design_window,
          time_window_for_eval=self.evaluation_window,
          matching_metrics={'response': 1.0},
          pairs=pairs)
      _, detailed_results = test_class2.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=100)

      for key in detailed_results.keys():
        self.assertTrue(
            np.array_equal(detailed_results[key]['estimate'],
                           expected_detailed_results[key]['estimate']))

  def testComplexDesignWithPairing(self):
    df = pd.read_csv(
        'trimmed_match/example_datasets/example_data_for_design.csv',
        parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date'])
    for col in ['response', 'cost', 'geo']:
      df[col] = pd.to_numeric(df[col])
    df.rename(columns={'cost': 'spend'}, inplace=True)

    test_class_no_pairing = TrimmedMatchGeoXDesign(
        GeoXType.HOLD_BACK,
        pretest_data=df,
        time_window_for_design=TimeWindow('2020-01-01', '2020-12-29'),
        time_window_for_eval=TimeWindow('2020-12-02', '2020-12-29'),
        matching_metrics={'response': 1.0, 'spend': 0.01},
        pairs=None)

    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue

      test_class_no_pairing._geox_type = geox_type
      expected_results, expected_detailed_results = test_class_no_pairing.report_candidate_designs(
          budget_list=[100000],
          iroas_list=[0],
          use_cross_validation=True,
          num_pairs_filtered_list=[8],
          num_simulations=100)

      geos = test_class_no_pairing.geo_level_eval_data.loc[
          test_class_no_pairing.geo_level_eval_data['pair'] > 8, 'geo']
      pairs = test_class_no_pairing.pairs.loc[
          test_class_no_pairing.pairs['pair'] > 8,
          ['geo1', 'geo2', 'pair']].reset_index(drop=True)
      test_class_pairing = TrimmedMatchGeoXDesign(
          geox_type,
          pretest_data=df[df['geo'].isin(geos)],
          time_window_for_design=TimeWindow('2020-01-01', '2020-12-29'),
          time_window_for_eval=TimeWindow('2020-12-02', '2020-12-29'),
          matching_metrics={'response': 1.0, 'spend': 0.01},
          pairs=pairs)
      results, detailed_results = test_class_pairing.report_candidate_designs(
          budget_list=[100000],
          iroas_list=[0],
          use_cross_validation=True,
          num_pairs_filtered_list=[0],
          num_simulations=100)
      self.assertEqual(results['rmse'][0], expected_results['rmse'][0])
      self.assertTrue(
          np.array_equal(
              detailed_results[(100000.0, 0.0, 0)]['estimate'],
              expected_detailed_results[(100000.0, 0.0, 8)]['estimate']))

  def testTooFewGeos(self):
    """Checks an error is raised with less than 10 pairs."""
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      self.test_class._geox_type = geox_type
      with self.assertRaisesRegex(
          ValueError,
          r'the number of geos to design an experiment must be >= 20, got 4'):
        self.test_class.report_candidate_designs(
            budget_list=[30, 40],
            iroas_list=[0, 2],
            use_cross_validation=True,
            num_pairs_filtered_list=[0, 1, 100],
            num_simulations=100)

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
          num_simulations=100)

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

  def testReportCandidateDesignWithNegativeIROAS(self):
    """Checks the calculation with negative values in iroas_list."""

    self.test_class._pretest_data = pd.concat(
        [self.test_class._pretest_data, self.add_pair], sort=False)
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      self.test_class._geox_type = geox_type

      with self.assertRaisesRegex(
          ValueError,
          r'All elements in iroas_list must have non-negative values, got \[-1\].'
      ):
        self.test_class.report_candidate_designs(
            budget_list=[30, 40],
            iroas_list=[0, -1, 2],
            use_cross_validation=True,
            num_pairs_filtered_list=[0, 1, 100],
            num_simulations=100)

  def testInsufficientNumberOfGeos(self):
    with self.assertRaisesRegex(
        ValueError,
        r'the number of geos to design an experiment must be >= 20, got 4'):
      self.test_class.report_candidate_designs(
          budget_list=[30, 40],
          iroas_list=[0, 2],
          use_cross_validation=True,
          num_pairs_filtered_list=[0, 1, 100],
          num_simulations=100)

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

  def testOutputCandidateDesignAssignments(self):
    """Check that the design in output is ok."""
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
    self.test_class.generate_balanced_assignment(
        num_pairs_filtered=0, base_seed=0)
    self.assertTrue(
        self.test_class.geo_level_eval_data.equals(
            pd.DataFrame({
                'geo':
                    list(range(0, 20)),
                'pair':
                    sorted(list(range(1, 11)) * 2),
                'response': [101 + 2 * x for x in range(0, 20)],
                'spend': [1 + 2 * x for x in range(0, 20)],
                'assignment': [
                    0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0
                ]
            })))

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
    self.test_class.generate_balanced_assignment(
        num_pairs_filtered=0, base_seed=0)
    default_ids = self.test_class.geo_level_eval_data.copy()
    self.test_class.generate_balanced_assignment(
        num_pairs_filtered=0, base_seed=0, group_control=2, group_treatment=1)
    specific_ids = self.test_class.geo_level_eval_data

    self.assertTrue(
        default_ids.drop('assignment', axis=1).equals(
            specific_ids.drop('assignment', axis=1)))
    self.assertTrue(
        np.array_equal(default_ids['assignment'].values,
                       2 - specific_ids['assignment'].values))

  def testOutputCandidateDesignWithMissingObservation(self):
    """Checks that no error is raised if missing observation are present."""
    # geo 51 does not have any observation in the evaluation period, which could
    # cause aa_test_data and sign_test_data to be different. Sign_test_data
    # would miss any geo that doesn't have observation in the most recent weeks.
    test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=pd.DataFrame({
            'date':
                pd.to_datetime(['2019-01-01', '2019-10-01'] * 51 +
                               ['2019-01-01']),
            'geo':
                sorted(list(range(50)) * 2) + [50, 50, 51],
            'response':
                list(range(100, 200)) + [.1, .1, .1],
            'spend':
                [1] * 103
        }),
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0})

    _ = test_class.report_candidate_designs(
        budget_list=[30],
        iroas_list=[0],
        use_cross_validation=True,
        num_pairs_filtered_list=[0],
        num_simulations=200)
    _ = test_class.output_chosen_design(num_pairs_filtered=0, base_seed=0)
    for index in range(26):
      self.assertSetEqual(
          set(test_class.geo_level_eval_data['assignment'][(2 * index):(
              2 * (index + 1))]), set([0, 1]))

  def testOutputCandidateDesignWithMissingObservationInPretest(self):
    """Checks that no error is raised if missing observation are present."""
    # geo 51 does not have any observation in the pretest period, which could
    # cause aa_test_data and sign_test_data to be different.
    test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=pd.DataFrame({
            'date':
                pd.to_datetime(['2019-01-01', '2019-10-01'] * 51 +
                               ['2019-10-01']),
            'geo':
                sorted(list(range(50)) * 2) + [50, 50, 51],
            'response':
                sorted(list(range(50)) * 2) + [0.1, 1.1, .1],
            'spend': [1] * 100 + [1, 2, 3]
        }),
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window,
        matching_metrics={'response': 1.0})
    _ = test_class.report_candidate_designs(
        budget_list=[30],
        iroas_list=[0],
        use_cross_validation=True,
        num_pairs_filtered_list=[0],
        num_simulations=200)
    _ = test_class.output_chosen_design(num_pairs_filtered=0, base_seed=0)
    for index in range(26):
      self.assertSetEqual(
          set(test_class.geo_level_eval_data['assignment'][(2 * index):(
              2 * (index + 1))]), set([0, 1]))

  def testOutputCandidateDesignWithMissingObservationDifferentPeriods(self):
    """Checks that no error is raised if missing observation are present."""
    # geo 51 does not have any observation in the evaluation period, which could
    # cause aa_test_data and sign_test_data to be different when the evaluation
    # period is different than the most recent weeks.
    test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=pd.DataFrame({
            'date':
                pd.to_datetime(['2019-01-01', '2019-10-01'] * 51 +
                               ['2019-10-01']),
            'geo':
                sorted(list(range(50)) * 2) + [50, 50, 51],
            'response':
                list(range(100, 200)) + [.1, .1, .1],
            'spend':
                [1] * 103
        }),
        time_window_for_design=self.design_window,
        time_window_for_eval=TimeWindow(
            pd.Timestamp('2019-01-01'), pd.Timestamp('2019-02-01')),
        matching_metrics={'response': 1.0})

    _ = test_class.report_candidate_designs(
        budget_list=[30],
        iroas_list=[0],
        use_cross_validation=True,
        num_pairs_filtered_list=[0],
        num_simulations=200)
    _ = test_class.output_chosen_design(num_pairs_filtered=0, base_seed=0)
    for index in range(26):
      self.assertSetEqual(
          set(test_class.geo_level_eval_data['assignment'][(2 * index):(
              2 * (index + 1))]), set([0, 1]))


if __name__ == '__main__':
  unittest.main()
