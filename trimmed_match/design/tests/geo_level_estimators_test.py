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
"""Tests for geo_level_estimators."""

import pandas as pd
from trimmed_match.design import common_classes
from trimmed_match.design import geo_level_estimators

import unittest

TREATMENT = common_classes.GeoAssignment.TREATMENT
CONTROL = common_classes.GeoAssignment.CONTROL
EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT
POST_EXPERIMENT = common_classes.ExperimentPeriod.POST_EXPERIMENT


class AnalysisResultTest(unittest.TestCase):

  def testString(self):
    result = geo_level_estimators.AnalysisResult(
        point_estimate=1.0,
        confidence_interval=geo_level_estimators.ConfidenceInterval(
            0.0, 2.0, 0.8),
        trim_rate=0.3,
        trim_rate_cost=0.4)
    self.assertEqual(str(result),
                     "estimate=1.00, ci_lower=0.00, ci_upper=2.00")
    expected_dict = {
        "point_estimate": 1.0,
        "conf_interval_lower": 0.0,
        "conf_interval_upper": 2.0,
        "conf_level": 0.8,
        "trim_rate": 0.3,
        "trim_rate_cost": 0.4,
    }
    self.assertDictEqual(expected_dict, result.to_dict())


class GeoLevelEstimatorsTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self.df = pd.DataFrame({
        "geo": [1, 2, 3, 4, 5, 6, 7, 8],
        "period": [EXPERIMENT] * 8,
        "pair": [1, 2, 3, 4, 1, 2, 3, 4],
        "assignment": [CONTROL] * 4 + [TREATMENT] * 4,
        "cost": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
        "response": [0, 0, 0, 0, 1.0, 1.0, 1.0, 3.0],
    })

  def assertAnalysisResultEqual(self, expected, actual):
    self.assertAlmostEqual(expected.point_estimate, actual.point_estimate)
    self.assertAlmostEqual(expected.confidence_interval.lower,
                           actual.confidence_interval.lower, delta=1e-6)
    self.assertAlmostEqual(expected.confidence_interval.upper,
                           actual.confidence_interval.upper, delta=1e-6)
    self.assertAlmostEqual(expected.trim_rate, actual.trim_rate, delta=1e-6)

  def test_trimmed_match_valueerror_missing_pair_column(self):
    df = pd.DataFrame()
    expected_message = "Pairing is required to analyze with trimmed match."
    with self.assertRaisesRegex(ValueError, expected_message):
      geo_level_estimators.TrimmedMatch().analyze(df)

  def test_trimmed_match_use_cooldown(self):
    df = self.df.copy()
    df.loc[df.pair == 4, "period"] = POST_EXPERIMENT
    estimator = geo_level_estimators.TrimmedMatch(trim_rate=0.0)
    self.assertAnalysisResultEqual(
        estimator.analyze(geoxts=df, use_cooldown=True),
        estimator.analyze(geoxts=self.df)
    )
    self.assertAnalysisResultEqual(
        estimator.analyze(geoxts=df, use_cooldown=False),
        estimator.analyze(geoxts=self.df[self.df.pair != 4])
    )

  def test_trimmed_match_no_trimming(self):
    results = geo_level_estimators.TrimmedMatch(trim_rate=0.0).analyze(self.df)
    self.assertAlmostEqual(results.point_estimate, 1.5)
    self.assertEqual(results.trim_rate, 0.0)

  def test_trimmed_match_with_trim_rate(self):
    results = geo_level_estimators.TrimmedMatch(trim_rate=0.25).analyze(self.df)
    expected_results = geo_level_estimators.AnalysisResult(
        point_estimate=1.0,
        confidence_interval=geo_level_estimators.ConfidenceInterval(
            lower=1.0,
            upper=1.0,
            confidence_level=geo_level_estimators.DEFAULT_CONFIDENCE_LEVEL
        ),
        trim_rate=0.25)
    self.assertAnalysisResultEqual(expected_results, results)

  def test_trimmed_match_max_trim_rate(self):
    results_trim_rate = geo_level_estimators.TrimmedMatch(
        trim_rate=0.25).analyze(self.df)
    results_max_trim_rate = geo_level_estimators.TrimmedMatch(
        max_trim_rate=0.25).analyze(self.df)
    self.assertAnalysisResultEqual(results_trim_rate, results_max_trim_rate)

  def test_trimmed_match_change_assignment(self):
    df_edited = self.df.copy()
    df_edited["assignment"] = df_edited.assignment.map({TREATMENT: 5,
                                                        CONTROL: 6})
    results = geo_level_estimators.TrimmedMatch().analyze(
        geoxts=df_edited, group_control=6, group_treatment=5)
    expected_results = geo_level_estimators.TrimmedMatch().analyze(self.df)
    self.assertAnalysisResultEqual(expected_results, results)

if __name__ == "__main__":
  unittest.main()
