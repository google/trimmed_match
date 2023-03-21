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
"""Tests for trimmed_match_post_analysis."""
from unittest import mock

import numpy as np
import pandas as pd
from trimmed_match.design import common_classes
from trimmed_match.post_analysis import trimmed_match_post_analysis

import unittest

TrimmedMatchData = trimmed_match_post_analysis.TrimmedMatchData

CONTROL = common_classes.GeoAssignment.CONTROL
TREATMENT = common_classes.GeoAssignment.TREATMENT
EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT
POST_EXPERIMENT = common_classes.ExperimentPeriod.POST_EXPERIMENT


class TrimmedMatchPostAnalysis(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.test_data = pd.DataFrame({
        'geo': [1, 1, 2, 2, 3, 3, 4, 4],
        'response': [10, 10, 20, 20, 30, 30, 40, 40],
        'cost': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        'pair': [1, 1, 1, 1, 2, 2, 2, 2],
        'assignment': ([CONTROL] * 2 + [TREATMENT] * 2) * 2,
        'period': [EXPERIMENT, POST_EXPERIMENT] * 4,
    })
    self.dataframe = pd.DataFrame({
        'geo': [1, 2, 3, 4, 5, 6, 7, 8],
        'response': [10, 11, 20, 30, 30, 33, 40, 48],
        'cost': [1.0, 2.0, 2.0, 7.0, 3.0, 5.0, 4.0, 9.0],
        'pair': [1, 1, 2, 2, 3, 3, 4, 4],
        'assignment': [CONTROL, TREATMENT] * 4,
        'period': [EXPERIMENT] * 8,
    })
    self.df = pd.DataFrame({
        'date': ['2020-10-09', '2020-10-10', '2020-10-11'] * 4,
        'geo': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'response': [10, 10, 10, 20, 20, 20, 30, 30, 30, 40, 40, 40],
        'cost': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
        'pair': [EXPERIMENT] * 6 + [POST_EXPERIMENT] * 6,
        'assignment': ([CONTROL] * 3 + [TREATMENT] * 3) * 2,
    })
    self.data = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        self.dataframe, exclude_cooldown=True)

  def testPrepareDataTestOnly(self):
    dt = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        self.test_data, exclude_cooldown=True)
    self.assertTupleEqual(
        dt, TrimmedMatchData(
            pair=[1, 2],
            treatment_response=[20, 40],
            control_response=[10, 30],
            treatment_cost=[2.0, 4.0],
            control_cost=[1.0, 3.0],
            epsilon=[],
        ))

  def testPrepareDataWithGroupNotation(self):
    # change control to 1 and treatment to 2
    self.test_data['assignment'] = np.where(
        self.test_data['assignment'] == TREATMENT, 2, 1)
    dt = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        self.test_data,
        exclude_cooldown=True,
        group_control=1,
        group_treatment=2)
    self.assertTupleEqual(
        dt, TrimmedMatchData(
            pair=[1, 2],
            treatment_response=[20, 40],
            control_response=[10, 30],
            treatment_cost=[2.0, 4.0],
            control_cost=[1.0, 3.0],
            epsilon=[],
        ))

  def testPrepareDataWithInverseNotation(self):
    # change control to 1 and treatment to 0
    self.test_data['assignment'] = ((CONTROL + TREATMENT) -
                                    self.test_data['assignment'])
    dt = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        self.test_data,
        exclude_cooldown=True,
        group_control=TREATMENT,
        group_treatment=CONTROL)
    self.assertTupleEqual(
        dt, TrimmedMatchData(
            pair=[1, 2],
            treatment_response=[20, 40],
            control_response=[10, 30],
            treatment_cost=[2.0, 4.0],
            control_cost=[1.0, 3.0],
            epsilon=[],
        ))

  def testPrepareDataTestAndCooldown(self):
    dt = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        self.test_data, exclude_cooldown=False)
    self.assertTupleEqual(
        dt, TrimmedMatchData(
            pair=[1, 2],
            treatment_response=[40, 80],
            control_response=[20, 60],
            treatment_cost=[4.0, 8.0],
            control_cost=[2.0, 6.0],
            epsilon=[],
        ))

  def testCalculateExperimentResults(self):

    delta_response = [
        self.data.treatment_response[x] - self.data.control_response[x]
        for x in range(len(self.data.treatment_response))
    ]
    delta_spend = [
        self.data.treatment_cost[x] - self.data.control_cost[x]
        for x in range(len(self.data.treatment_response))
    ]
    estimate = sum(delta_response[2:]) / sum(delta_spend[2:])
    trim_rate = 0.25
    confidence = 0.9
    conf_interval_low = -29.374179
    conf_interval_up = 1.618974
    epsilons = [
        delta_response[x] - delta_spend[x] * estimate
        for x in range(len(delta_response))
    ]
    trimmed_pairs_indices = [0, 1]
    trimmed_pairs = [self.data.pair[x] for x in trimmed_pairs_indices]
    incremental_cost = 13.0
    incremental_response = 13.0 * estimate
    incremental_response_lower = incremental_cost * conf_interval_low
    incremental_response_upper = incremental_cost * conf_interval_up
    treatment_response = 122.0
    control_response = 100.0
    results = trimmed_match_post_analysis.calculate_experiment_results(
        self.data, max_trim_rate=0.25, confidence=0.9, trim_rate=-1)

    fit = results.report
    self.assertAlmostEqual(fit.estimate, estimate, places=3)
    self.assertAlmostEqual(fit.trim_rate, trim_rate, places=3)
    self.assertAlmostEqual(fit.confidence, confidence, places=3)
    self.assertAlmostEqual(fit.conf_interval_low, conf_interval_low, places=3)
    self.assertAlmostEqual(fit.conf_interval_up, conf_interval_up, places=3)
    self.assertListEqual(results.trimmed_pairs, trimmed_pairs)
    self.assertTrue(np.allclose(results.data.epsilon, epsilons, atol=1e-5))
    self.assertAlmostEqual(incremental_cost, results.incremental_cost, places=3)
    self.assertAlmostEqual(
        incremental_response, results.incremental_response, places=3)
    self.assertAlmostEqual(
        incremental_response_lower,
        results.incremental_response_lower,
        places=3)
    self.assertAlmostEqual(
        incremental_response_upper,
        results.incremental_response_upper,
        places=3)
    self.assertAlmostEqual(
        treatment_response, results.treatment_response, places=3)
    self.assertAlmostEqual(
        control_response, results.control_response, places=3)

  def testCalculateExperimentResultsNegativeIncrementalCost(self):
    """Checks that the CI for incremental response is correct with negative incremental cost."""
    dataframe = self.dataframe.copy()
    # flip assignment
    dataframe['assignment'] = (CONTROL + TREATMENT) - dataframe['assignment']
    data = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        dataframe, exclude_cooldown=True)
    delta_response = [
        data.treatment_response[x] - data.control_response[x]
        for x in range(len(data.treatment_response))
    ]
    delta_spend = [
        data.treatment_cost[x] - data.control_cost[x]
        for x in range(len(data.treatment_response))
    ]
    estimate = sum(delta_response[2:]) / sum(delta_spend[2:])
    trim_rate = 0.25
    confidence = 0.9
    conf_interval_low = -29.374179
    conf_interval_up = 1.618974
    epsilons = [
        delta_response[x] - delta_spend[x] * estimate
        for x in range(len(delta_response))
    ]
    trimmed_pairs_indices = [0, 1]
    trimmed_pairs = [data.pair[x] for x in trimmed_pairs_indices]
    incremental_cost = - 13.0
    incremental_response = incremental_cost * estimate
    incremental_response_lower = incremental_cost * conf_interval_up
    incremental_response_upper = incremental_cost * conf_interval_low
    treatment_response = 100.0
    control_response = 122.0
    results = trimmed_match_post_analysis.calculate_experiment_results(
        data, max_trim_rate=0.25, confidence=0.9, trim_rate=-1)

    fit = results.report
    self.assertAlmostEqual(fit.estimate, estimate, places=3)
    self.assertAlmostEqual(fit.trim_rate, trim_rate, places=3)
    self.assertAlmostEqual(fit.confidence, confidence, places=3)
    self.assertAlmostEqual(fit.conf_interval_low, conf_interval_low, places=3)
    self.assertAlmostEqual(fit.conf_interval_up, conf_interval_up, places=3)
    self.assertListEqual(results.trimmed_pairs, trimmed_pairs)
    self.assertTrue(np.allclose(results.data.epsilon, epsilons, atol=1e-5))
    self.assertAlmostEqual(incremental_cost, results.incremental_cost, places=3)
    self.assertAlmostEqual(
        incremental_response, results.incremental_response, places=3)
    self.assertAlmostEqual(
        incremental_response_lower,
        results.incremental_response_lower,
        places=3)
    self.assertAlmostEqual(
        incremental_response_upper,
        results.incremental_response_upper,
        places=3)
    self.assertAlmostEqual(
        treatment_response, results.treatment_response, places=3)
    self.assertAlmostEqual(
        control_response, results.control_response, places=3)

  def testPrepareDataCorrectGroupLabels(self):
    with self.assertRaisesRegex(
        ValueError,
        r'The data do not have observations for the two groups.' +
        r'The labels found in the data in input are ' +
        fr'{set(self.dataframe.assignment.values)}, and the expected ' +
        r'labels are: Treatment=10, Control=20'):
      trimmed_match_post_analysis.prepare_data_for_post_analysis(
          self.dataframe, group_treatment=10, group_control=20)

  def testUnequalGroupSizes(self):
    temp_df = self.dataframe.copy()
    # remove geo #2 which is in treatment
    temp_df = temp_df[temp_df['geo'] != 2]
    with self.assertRaisesRegex(ValueError,
                                r'Some pairs do not have one geo for ' +
                                r'each group.'):
      trimmed_match_post_analysis.prepare_data_for_post_analysis(
          temp_df, exclude_cooldown=True)

    temp_df = self.dataframe.copy()
    # remove geo #2 and #3 which are in control
    temp_df = temp_df[~temp_df['geo'].isin([1, 3])]
    with self.assertRaisesRegex(ValueError,
                                r'Some pairs do not have one geo for ' +
                                r'each group.'):
      trimmed_match_post_analysis.prepare_data_for_post_analysis(
          temp_df, exclude_cooldown=True)

  def testGeosPerPair(self):
    temp_df = self.dataframe.copy()
    # reassigne geo #2 which is in treatment, and geo #3 which is control
    temp_df.loc[temp_df['geo'] == 2, 'assignment'] = 0
    temp_df.loc[temp_df['geo'] == 3, 'assignment'] = 1
    with self.assertRaisesRegex(
        ValueError, r'Some pairs do not have one geo for each group.'):
      trimmed_match_post_analysis.prepare_data_for_post_analysis(
          temp_df, exclude_cooldown=True)

  def testManyGeosPerPairWithSameAssignment(self):
    temp_df = self.dataframe.copy()
    # add one additional geo to pairs #1 and #4
    temp_df = temp_df.append(pd.DataFrame({
        'geo': [9, 10],
        'response': [10, 11],
        'cost': [1.0, 2.0],
        'pair': [1, 4],
        'assignment': [0, 1],
        'period': [1, 1],
    }))
    with self.assertRaisesRegex(
        ValueError, r'Some pairs do not have one geo for each group.'):
      trimmed_match_post_analysis.prepare_data_for_post_analysis(
          temp_df, exclude_cooldown=True)

  def testDuplicateGeo(self):
    temp_df = self.dataframe.copy()
    # change geo #4 to geo #2, so that geo #2 is duplicated
    temp_df.loc[temp_df['geo'] == 4, 'geo'] = 2
    # change geo #5 to geo #3, so that geo #3 is duplicated
    temp_df.loc[temp_df['geo'] == 5, 'geo'] = 3
    with self.assertRaisesRegex(
        ValueError, r'Some geos are duplicated and appear in multiple pairs.'):
      trimmed_match_post_analysis.prepare_data_for_post_analysis(
          temp_df, exclude_cooldown=True)

  def testReportExperimentResults(self):
    results = trimmed_match_post_analysis.calculate_experiment_results(
        self.data, max_trim_rate=0.25, confidence=0.9, trim_rate=-1)
    with mock.patch(
        'builtins.print', autospec=True, side_effect=print) as mock_print:
      trimmed_match_post_analysis.report_experiment_results(results, 1)

      self.assertEqual(mock_print.call_args_list[0][0][0],
                       'Summary of the results for the iROAS:\n\n')
      self.assertEqual(
          mock_print.call_args_list[1][0][0],
          'estimate\t trim_rate\t ci_level\t confidence interval')
      self.assertEqual(
          mock_print.call_args_list[2][0][0],
          '1.571\t\t 0.25\t\t 0.90\t\t [-29.374, 1.619]\n\n')
      self.assertEqual(mock_print.call_args_list[3][0][0], 'cost = 13')
      self.assertEqual(mock_print.call_args_list[4][0][0],
                       '\nincremental response = 20.4')
      self.assertEqual(mock_print.call_args_list[5][0][0],
                       '\ntreatment response = 122')
      self.assertEqual(
          mock_print.call_args_list[6][0][0],
          '\nincremental response as % of treatment response = 16.74%\n')

  def testCheckInputData(self):
    temp_df = self.df.copy()
    # remove one observation for geo #2
    temp_df = temp_df[~((temp_df['geo'] == 2) &
                        (temp_df['date'] == '2020-10-10'))]
    geox_data = trimmed_match_post_analysis.check_input_data(temp_df)
    expected_df = pd.DataFrame({
        'date': ['2020-10-09', '2020-10-10', '2020-10-11'] * 4,
        'geo': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'pair': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'assignment': ([CONTROL] * 3 + [TREATMENT] * 3) * 2,
        'response': [10, 10, 10, 20, 0.0, 20, 30, 30, 30, 40, 40, 40],
        'cost': [1.0, 1.0, 1.0, 2.0, 0.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
    }).sort_values(by=['date', 'geo']).reset_index(drop=True)
    self.assertTrue(geox_data.equals(expected_df))

  def testCheckInputDataOrder(self):
    temp = pd.DataFrame({
        'date': pd.date_range('2021-01-01', periods=4),
        'pair': [2, 2, 1, 1],
        'assignment': [1, 2, 2, 1],
        'geo': [1, 2, 3, 4],
        'response': 0,
        'cost': 0
    })
    geox_data = trimmed_match_post_analysis.check_input_data(
        temp, group_control=1, group_treatment=2)
    self.assertTrue(
        np.array_equal(
            geox_data[['date', 'geo', 'pair', 'assignment']],
            pd.DataFrame({
                'date': np.repeat(temp['date'], 4),
                'geo': [1, 2, 3, 4] * 4,
                'pair': [2, 2, 1, 1] * 4,
                'assignment': [1, 2, 2, 1] * 4
            })))

  def testCheckInputDataColumns(self):
    temp_df = self.df.copy()
    # remove the column assignment
    temp_df.drop(columns='assignment', inplace=True)
    with self.assertRaisesRegex(
        ValueError,
        'The mandatory columns {\'assignment\'} are missing from the input data'
    ):
      trimmed_match_post_analysis.check_input_data(temp_df)

  def testCheckInputDataCorrectGroupLabels(self):
    with self.assertRaisesRegex(
        ValueError,
        r'The data do not have observations for the two groups.' +
        r'Check the data and the values used to indicate the ' +
        r'assignments for treatment and control. The labels ' +
        r'found in the data in input are ' +
        fr'\[{CONTROL}, {TREATMENT}\], and the expected labels ' +
        r'are: Treatment=10, Control=20'):
      trimmed_match_post_analysis.check_input_data(
          self.df, group_control=20, group_treatment=10)

  def testCheckUnequalGroupSizesInInputData(self):
    temp_df = self.df.copy()
    # remove geo #2 which is in treatment
    temp_df = temp_df[temp_df['geo'] != 2]
    with self.assertRaisesRegex(ValueError,
                                r'Some pairs do not have one geo for ' +
                                r'each group.'):
      trimmed_match_post_analysis.check_input_data(temp_df)

    temp_df = self.df.copy()
    # remove geo #1 which is in control
    temp_df = temp_df[temp_df['geo'] != 1]
    with self.assertRaisesRegex(ValueError,
                                r'Some pairs do not have one geo for ' +
                                r'each group.'):
      trimmed_match_post_analysis.check_input_data(temp_df)

  def testCheckGeosPerPairInInputData(self):
    temp_df = self.df.copy()
    # reassign geo #2 which is in treatment, and geo #3 which is control
    temp_df.loc[temp_df['geo'] == 2, 'assignment'] = CONTROL
    temp_df.loc[temp_df['geo'] == 3, 'assignment'] = TREATMENT
    with self.assertRaisesRegex(
        ValueError, r'Some pairs do not have one geo for each group.'):
      trimmed_match_post_analysis.check_input_data(temp_df)

  def testManyGeosPerPairWithSameAssignmentInInputData(self):
    temp_df = self.df.copy()
    # add one additional geo to pairs #1 and #4
    temp_df = temp_df.append(pd.DataFrame({
        'date': ['2020-10-10', '2020-10-10'],
        'geo': [10, 9],
        'response': [10, 11],
        'cost': [1.0, 2.0],
        'pair': [1, 4],
        'assignment': [CONTROL, TREATMENT],
        'period': [1, 1],
    }))
    with self.assertRaisesRegex(
        ValueError, r'Some pairs do not have one geo for each group.'):
      trimmed_match_post_analysis.check_input_data(temp_df)

  def testDuplicateGeoInInputData(self):
    temp_df = self.df.copy()
    # change geo #4 to geo #2, so that geo #2 is duplicated
    temp_df.loc[temp_df['geo'] == 4, 'geo'] = 2
    # change geo #5 to geo #3, so that geo #3 is duplicated
    temp_df.loc[temp_df['geo'] == 5, 'geo'] = 3
    with self.assertRaisesRegex(
        ValueError, r'Some geos are duplicated and appear in multiple pairs.'):
      trimmed_match_post_analysis.check_input_data(temp_df)

  def testGeosNotInExperimentAreExcluded(self):
    temp_df = self.df.copy()
    # add two additional geos with assignment -1 in new and different pairs.
    temp_df = temp_df.append(pd.DataFrame({
        'date': ['2020-10-10', '2020-10-10'],
        'geo': [9, 10],
        'response': [10, 11],
        'cost': [1.0, 2.0],
        'pair': [100, 101],
        'assignment': [-1, -1],
    }))
    geox_data = trimmed_match_post_analysis.check_input_data(temp_df)
    expected_df = pd.DataFrame({
        'date': ['2020-10-09', '2020-10-10', '2020-10-11'] * 4,
        'geo': [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        'pair': [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        'assignment': ([CONTROL] * 3 + [TREATMENT] * 3) * 2,
        'response': [10, 10, 10, 20, 20.0, 20, 30, 30, 30, 40, 40, 40],
        'cost': [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0],
    }).sort_values(by=['date', 'geo']).reset_index(drop=True)
    self.assertTrue(geox_data.reset_index(drop=True).equals(expected_df))

  def testDuplicateGeoInSamePair(self):
    temp_df = self.df.copy()
    # change geo #1 to geo #2, so that geo #2 is duplicated in pair 1 with
    # assignment 0 and 1
    temp_df.loc[temp_df['geo'] == 1, 'geo'] = 2
    with self.assertRaisesRegex(
        ValueError, r'Some geos are duplicated and appear in multiple pairs.'):
      trimmed_match_post_analysis.check_input_data(temp_df)

if __name__ == '__main__':
  unittest.main()
