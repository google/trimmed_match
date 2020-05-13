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

import collections
from unittest import mock
import numpy as np
import pandas as pd

from trimmed_match.post_analysis import trimmed_match_post_analysis
import unittest


TrimmedMatchData = collections.namedtuple('TrimmedMatchData', [
    'pair', 'treatment_response', 'control_response', 'treatment_cost',
    'control_cost', 'epsilon'
])


class TrimmedMatchPostAnalysis(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super().setUp()
    self.test_data = pd.DataFrame({
        'geo': [1, 1, 2, 2, 3, 3, 4, 4],
        'response': [10, 10, 20, 20, 30, 30, 40, 40],
        'cost': [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0],
        'pair': [1, 1, 1, 1, 2, 2, 2, 2],
        'assignment': [0, 0, 1, 1, 0, 0, 1, 1],
        'period': [1, 2, 1, 2, 1, 2, 1, 2],
    })
    self.dataframe = pd.DataFrame({
        'geo': [1, 2, 3, 4, 5, 6, 7, 8],
        'response': [10, 11, 20, 30, 30, 33, 40, 48],
        'cost': [1.0, 2.0, 2.0, 7.0, 3.0, 5.0, 4.0, 9.0],
        'pair': [1, 1, 2, 2, 3, 3, 4, 4],
        'assignment': [0, 1, 0, 1, 0, 1, 0, 1],
        'period': [1, 1, 1, 1, 1, 1, 1, 1],
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
    std_error = 0.041
    trim_rate = 0.25
    confidence = 0.9
    conf_interval_low = -29.374
    conf_interval_up = 1.619
    epsilons = [
        delta_response[x] - delta_spend[x] * estimate
        for x in range(len(delta_response))
    ]
    trimmed_pairs_indices = [0, 1]
    trimmed_pairs = [self.data.pair[x] for x in trimmed_pairs_indices]
    incremental_cost = 13.0
    lift = 13.0 * estimate
    treatment_response = 122.0
    results = trimmed_match_post_analysis.calculate_experiment_results(
        self.data, max_trim_rate=0.25, confidence=0.9, trim_rate=-1)

    fit = results.report
    self.assertAlmostEqual(fit.estimate, estimate, places=3)
    self.assertAlmostEqual(fit.std_error, std_error, places=3)
    self.assertAlmostEqual(fit.trim_rate, trim_rate, places=3)
    self.assertAlmostEqual(fit.confidence, confidence, places=3)
    self.assertAlmostEqual(fit.conf_interval_low, conf_interval_low, places=3)
    self.assertAlmostEqual(fit.conf_interval_up, conf_interval_up, places=3)
    self.assertListEqual(results.trimmed_pairs, trimmed_pairs)
    self.assertTrue(np.allclose(results.data.epsilon, epsilons, atol=1e-5))
    self.assertAlmostEqual(incremental_cost, results.incremental_cost, places=3)
    self.assertAlmostEqual(lift, results.lift, places=3)
    self.assertAlmostEqual(
        treatment_response, results.treatment_response, places=3)

  def testReportExperimentResults(self):
    results = trimmed_match_post_analysis.calculate_experiment_results(
        self.data, max_trim_rate=0.25, confidence=0.9, trim_rate=-1)
    with mock.patch(
        'builtins.print', autospec=True, side_effect=print) as mock_print:
      trimmed_match_post_analysis.report_experiment_results(results)

      self.assertEqual(mock_print.call_args_list[0][0][0],
                       'Summary of the results for the iROAS:\n\n')
      self.assertEqual(
          mock_print.call_args_list[1][0][0],
          'estimate\t std. error\t trim_rate\t ci_level\t confidence interval')
      self.assertEqual(
          mock_print.call_args_list[2][0][0],
          '1.571\t\t 0.041\t\t 0.25\t\t 0.90\t\t [-29.374, 1.619]\n\n')
      self.assertEqual(mock_print.call_args_list[3][0][0], 'cost = 13')
      self.assertEqual(mock_print.call_args_list[4][0][0],
                       '\nincremental response = 20.4')
      self.assertEqual(mock_print.call_args_list[5][0][0],
                       '\ntreatment response = 122')
      self.assertEqual(
          mock_print.call_args_list[6][0][0],
          '\nincremental response as % of treatment response = 16.74%\n')


if __name__ == '__main__':
  unittest.main()
