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
# Lint: python3
"""Tests for ads.amt.geox.trimmed_match.design."""
from absl.testing import parameterized
import numpy as np
import pandas as pd

from trimmed_match.design import common_classes
from trimmed_match.design import matched_pairs_rmse
import unittest

GeoLevelData = common_classes.GeoLevelData
GeoXType = common_classes.GeoXType
GeoLevelPotentialOutcomes = common_classes.GeoLevelPotentialOutcomes
MatchedPairsRMSE = matched_pairs_rmse.MatchedPairsRMSE

CONTROL = common_classes.GeoAssignment.CONTROL
TREATMENT = common_classes.GeoAssignment.TREATMENT
EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT
PRE_EXPERIMENT = common_classes.ExperimentPeriod.PRE_EXPERIMENT


class ConstructPotentialOutcomesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._geox_eval_data = pd.DataFrame({
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "spend": [10.0, 20.0]
    })
    self._budget = 30
    self._hypothesized_iroas = 1

  def testGoDark(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.GO_DARK, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = pd.DataFrame({
        "pair": [0, 0],
        "geo": [1, 2],
        "spend_control": [20.0, 40.0],
        "spend_treatment": [0.0, 0.0],
        "response_control": [10.0, 20.0],
        "response_treatment": [0.0, 0.0],
    })
    pd.testing.assert_frame_equal(potential_outcomes, expected)

  def testGoDarkWithHeavyUpControl(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = pd.DataFrame({
        "pair": [0, 0],
        "geo": [1, 2],
        "spend_control": [20.0, 40.0],
        "spend_treatment": [0.0, 0.0],
        "response_control": [20.0, 40.0],
        "response_treatment": [0.0, 0.0],
    })
    pd.testing.assert_frame_equal(potential_outcomes, expected)

  def testHeavyUp(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.HEAVY_UP, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = pd.DataFrame({
        "pair": [0, 0],
        "geo": [1, 2],
        "spend_control": [10.0, 20.0],
        "spend_treatment": [30.0, 60.0],
        "response_control": [10.0, 20.0],
        "response_treatment": [30.0, 60.0],
    })
    pd.testing.assert_frame_equal(potential_outcomes, expected)

  def testHeavyDown(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.HEAVY_DOWN, self._geox_eval_data, self._budget,
        self._hypothesized_iroas)
    expected = pd.DataFrame({
        "pair": [0, 0],
        "geo": [1, 2],
        "spend_control": [10.0, 20.0],
        "spend_treatment": [0.0, 0.0],
        "response_control": [10.0, 20.0],
        "response_treatment": [0.0, 0.0],
    })
    pd.testing.assert_frame_equal(potential_outcomes, expected)

  def testHoldBack(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.HOLD_BACK, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = pd.DataFrame({
        "pair": [0, 0],
        "geo": [1, 2],
        "spend_control": [0.0, 0.0],
        "spend_treatment": [20.0, 40.0],
        "response_control": [10.0, 20.0],
        "response_treatment": [30.0, 60.0],
    })
    pd.testing.assert_frame_equal(potential_outcomes, expected)

  def testUnknownGeoXType(self):
    """Checks an error is raised if the GeoX type is unknown."""
    with self.assertRaisesRegex(ValueError, "Unknown geox_type: \'UNKNOWN\'"):
      matched_pairs_rmse._construct_potential_outcomes(
          "UNKNOWN", self._geox_eval_data,
          (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
          self._hypothesized_iroas)


class MatchedPairsRMSETest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._geo_pairs_eval_data = pd.DataFrame({
        "geo": [1, 2, 3, 4],
        "pair": [1, 1, 2, 2],
        "response": [10, 20, 30, 40],
        "spend": [10, 20, 30, 40]
    })
    self._perfect_geo_pairs_eval_data = pd.DataFrame({
        "geo": [1, 2, 3, 4],
        "pair": [1, 1, 2, 2],
        "response": [10, 10, 30, 30],
        "spend": [10, 10, 30, 30]
    })
    self._larger_geo_pairs_data = pd.DataFrame({
        "geo": [1, 2, 3, 4, 5, 6, 7, 8],
        "response": [10.0, 11.0, 20.0, 30.0, 30.0, 33.0, 40.0, 48.0],
        "spend": [1.0, 2.0, 3.0, 7.0, 3.0, 5.0, 4.0, 9.0],
        "pair": [1, 1, 2, 2, 3, 3, 4, 4],
    })
    self._geo_pairs_period_data = pd.DataFrame({
        "geo": [1, 2, 3, 4, 1, 2, 3, 4],
        "date": pd.to_datetime(["2020-01-01"] * 4 + ["2020-01-02"] * 4),
        "period": [EXPERIMENT] * 4 + [PRE_EXPERIMENT] * 4,
        "response": [10.0, 11.0, 20.0, 30.0, 30.0, 33.0, 40.0, 48.0],
        "spend": [1.0, 2.0, 3.0, 7.0, 3.0, 5.0, 4.0, 9.0],
        "pair": [1, 1, 2, 2, 1, 1, 2, 2],
    })
    self._budget = 10
    self._hypothesized_iroas = 1

    # TODO(b/147698415): adding a more complex test example here

  def testHypothesizedIroasNegative(self):
    """Checks an error is raised if the hypothesized iROAS is negative."""
    with self.assertRaisesRegex(ValueError, "iROAS must be positive, got -1.0"):
      MatchedPairsRMSE(GeoXType.GO_DARK, self._geo_pairs_eval_data,
                       self._budget, -1.0)

  def testMissingColumns(self):
    """Checks an error is raised if one of the required columns is missing."""
    for col in ["geo", "pair", "response", "spend"]:
      geo_pairs_eval_data = self._geo_pairs_eval_data.copy()
      geo_pairs_eval_data.pop(col)
      with self.assertRaisesRegex(
          ValueError, fr"Missing {set([col])} column in geo_pairs_eval_data"):
        MatchedPairsRMSE(GeoXType.GO_DARK, geo_pairs_eval_data, self._budget,
                         self._hypothesized_iroas)

  def testDuplicatedGeos(self):
    """Checks an error is raised if geos are duplicated."""
    geo_pairs_eval_data = self._geo_pairs_eval_data.copy()
    geo_pairs_eval_data.loc[geo_pairs_eval_data["geo"] == 3, "geo"] = 1
    with self.assertRaisesRegex(
        ValueError, r"Duplicated values in geo_pairs_eval_data"):
      MatchedPairsRMSE(GeoXType.GO_DARK, geo_pairs_eval_data, self._budget,
                       self._hypothesized_iroas)

  def testMissingGeoPeriods(self):
    """Checks an error is raised if geo-periods are missing."""
    geo_pairs_eval_data = self._geo_pairs_eval_data.copy()
    geo_pairs_eval_data["period"] = EXPERIMENT
    geo_pairs_eval_data.loc[geo_pairs_eval_data["geo"] == 3, "period"] = 0
    with self.assertRaisesRegex(
        ValueError, r"Missing values in geo_pairs_eval_data"):
      MatchedPairsRMSE(GeoXType.GO_DARK, geo_pairs_eval_data, self._budget,
                       self._hypothesized_iroas)

  def testMissingGeoDate(self):
    """Checks an error is raised if a geo-date is missing."""
    geo_pairs_period_data = self._geo_pairs_period_data.copy()
    geo_pairs_period_data.drop(3, inplace=True)
    with self.assertRaisesRegex(
        ValueError, r"Missing values in geo_pairs_eval_data"):
      MatchedPairsRMSE(GeoXType.GO_DARK, geo_pairs_period_data, self._budget,
                       self._hypothesized_iroas)

  def testMissingExperimentValue(self):
    """Checks an error is raised if geo-periods are duplicated."""
    geo_pairs_eval_data = self._geo_pairs_eval_data.copy()
    geo_pairs_eval_data["period"] = -5
    with self.assertRaisesRegex(ValueError,
                                fr"Missing experiment value {EXPERIMENT} in " +
                                r"geo_pairs_eval_data.period"):
      MatchedPairsRMSE(GeoXType.GO_DARK, geo_pairs_eval_data, self._budget,
                       self._hypothesized_iroas)

  def testGeosNotPairedProperly(self):
    """Checks an error is raised if geos are not paired properly."""
    geo_pairs_eval_data = self._geo_pairs_eval_data.copy()
    geo_pairs_eval_data.loc[geo_pairs_eval_data["geo"] == 3, "pair"] = 1
    with self.assertRaisesRegex(
        KeyError, "Geos in geo_pairs_eval_data are not paired properly"):
      MatchedPairsRMSE(GeoXType.GO_DARK, geo_pairs_eval_data, self._budget,
                       self._hypothesized_iroas)

  def testSimulateGeoXDataShuffledData(self):
    dataframe = self._larger_geo_pairs_data.copy()
    base_seed = 1000
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr_exp_ordered = MatchedPairsRMSE(
          geox_type=geox_type,
          geo_pairs_eval_data=dataframe,
          budget=10.0,
          hypothesized_iroas=1.0,
          base_seed=base_seed)
      mpr_exp_shuffled = MatchedPairsRMSE(
          geox_type=geox_type,
          geo_pairs_eval_data=dataframe.sample(
              frac=1, random_state=0).reset_index(drop=True),
          budget=10.0,
          hypothesized_iroas=1.0,
          base_seed=base_seed)
      pd.util.testing.assert_frame_equal(mpr_exp_ordered._geox_data,
                                         mpr_exp_shuffled._geox_data)

  def testSimulateGeoXDataSamePreExperimentPeriod(self):
    dataframe = self._geo_pairs_period_data.copy()
    base_seed = 1000
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr_exp = MatchedPairsRMSE(
          geox_type=geox_type,
          geo_pairs_eval_data=dataframe,
          budget=10.0,
          hypothesized_iroas=1.0,
          base_seed=base_seed)
      geo_pairs_preexperiment_data = mpr_exp._simulate_geox_data(0).query(
          f"period == {PRE_EXPERIMENT}").copy()[[
              "geo", "pair", "spend", "response"]]
      dataframe_preexperiment = dataframe.query(f"period == {PRE_EXPERIMENT}")[
          ["geo", "pair", "spend", "response"]]
      pd.util.testing.assert_frame_equal(
          geo_pairs_preexperiment_data.reset_index(drop=True),
          dataframe_preexperiment.reset_index(drop=True))

  def testSimulateGeoXDataRandomization(self):
    """Checks randomization within the pair."""
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr = MatchedPairsRMSE(geox_type, self._geo_pairs_eval_data,
                             self._budget, self._hypothesized_iroas)
      geox_data = mpr._simulate_geox_data(0)
      self.assertSetEqual(set(mpr._potential_outcomes.pair.values),
                          set(geox_data.pair.values))
      self.assertSetEqual(set(mpr._potential_outcomes.geo.values),
                          set(geox_data.geo.values))
      np.testing.assert_array_equal(geox_data.assignment,
                                    [CONTROL, TREATMENT, CONTROL, TREATMENT])

  def testSimulatedGeoXDataValue(self):
    """Checks the data accuracy."""
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr = MatchedPairsRMSE(geox_type, self._geo_pairs_eval_data, self._budget,
                             self._hypothesized_iroas)
      outcomes = mpr._potential_outcomes
      geox_data = mpr._simulate_geox_data(0)
      np.testing.assert_array_equal(
          outcomes.loc[geox_data.assignment == CONTROL,
                       ["response_control", "spend_control"]].values,
          geox_data.loc[geox_data.assignment == CONTROL,
                        ["response", "spend"]].values,
      )
      np.testing.assert_array_equal(
          outcomes.loc[geox_data.assignment == TREATMENT,
                       ["response_treatment", "spend_treatment"]].values,
          geox_data.loc[geox_data.assignment == TREATMENT,
                        ["response", "spend"]].values,
      )

  def testReportValueError(self):
    mpr = MatchedPairsRMSE(
        GeoXType.HOLD_BACK,
        self._geo_pairs_eval_data,
        self._budget,
        self._hypothesized_iroas,
        base_seed=1000)

    @parameterized.parameters((-0.1, -0.2), (0.5, 0.1), (0.25, 0.3))
    def _(self, max_trim_rate, trim_rate):
      with self.assertRaises(ValueError):
        mpr.report(1, max_trim_rate, trim_rate)

  def testReportPerfectiROAS(self):
    """Checks the calculation with zero RMSE."""
    for geox_type in GeoXType:
      if geox_type in [GeoXType.HOLD_BACK, GeoXType.CONTROL, GeoXType.GO_DARK]:
        continue
      mpr = MatchedPairsRMSE(
          geox_type,
          self._geo_pairs_eval_data,
          self._budget,
          self._hypothesized_iroas,
          base_seed=1000)
      (report, _) = mpr.report(num_simulations=100, trim_rate=0.0)
      self.assertEqual(0.0, report)

  def testReportPerfectPairs(self):
    """Checks the calculation with perfect pairs."""
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr = MatchedPairsRMSE(
          geox_type,
          self._perfect_geo_pairs_eval_data,
          self._budget,
          0.0,
          base_seed=1000)
      report, _ = mpr.report(num_simulations=100, trim_rate=0.0)
      self.assertEqual(0.0, report)

  def testReportNoisy(self):
    """Checks the calculation with nonzero RMSE."""
    mpr = MatchedPairsRMSE(
        GeoXType.HOLD_BACK,
        self._geo_pairs_eval_data,
        self._budget,
        self._hypothesized_iroas,
        base_seed=100000)
    (report, _) = mpr.report(num_simulations=100, trim_rate=0.0)
    self.assertAlmostEqual(1.5, report, delta=0.1)

  def testReportNoisyDifferentGeoOrder(self):
    """Checks the calculation with nonzero RMSE when geo_pairs_eval_data order is changed."""
    mpr = MatchedPairsRMSE(
        GeoXType.HOLD_BACK,
        self._geo_pairs_eval_data,
        self._budget,
        self._hypothesized_iroas,
        base_seed=100000)
    (report, _) = mpr.report(num_simulations=100)
    mpr_sorted = MatchedPairsRMSE(
        GeoXType.HOLD_BACK,
        self._geo_pairs_eval_data.sort_values(
            by=["pair", "geo"], ascending=[True, False]),
        self._budget,
        self._hypothesized_iroas,
        base_seed=100000)
    (report_sorted, _) = mpr_sorted.report(num_simulations=100)

    self.assertAlmostEqual(
        abs(report - report_sorted) / report_sorted, 0, delta=0.00001)

  def testReportTrimmedPairs(self):
    """Checks the reported trimmed pairs in a simulation."""
    dataframe = self._larger_geo_pairs_data.copy()
    base_seed = 1000
    trimmed_pairs = {
        GeoXType.GO_DARK: [2, 3],
        GeoXType.HOLD_BACK: [2, 3],
        GeoXType.HEAVY_UP: [3, 4],
        GeoXType.HEAVY_DOWN: [3, 4],
        GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL: [2, 3],
    }
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr = MatchedPairsRMSE(
          geox_type=geox_type,
          geo_pairs_eval_data=dataframe,
          budget=1.0,
          hypothesized_iroas=0.0,
          base_seed=base_seed)
      _, report = mpr.report(num_simulations=1, trim_rate=0.0)
      self.assertFalse(report.trimmed_pairs.values[0])
      _, report = mpr.report(
          num_simulations=1, trim_rate=0.25, max_trim_rate=0.25)

      self.assertCountEqual(report.trimmed_pairs.values[0],
                            trimmed_pairs[geox_type])

  def testResultsSameWithAndWithoutPreExperimentPeriod(self):
    dataframe = self._geo_pairs_period_data.copy()
    base_seed = 1000
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr_exp = MatchedPairsRMSE(
          geox_type=geox_type,
          geo_pairs_eval_data=dataframe[dataframe["period"] == EXPERIMENT],
          budget=10.0,
          hypothesized_iroas=1.0,
          base_seed=base_seed)
      rmse_exp, report_exp = mpr_exp.report(num_simulations=1, trim_rate=0.0)
      mpr_all = MatchedPairsRMSE(
          geox_type=geox_type,
          geo_pairs_eval_data=dataframe,
          budget=10.0,
          hypothesized_iroas=1.0,
          base_seed=base_seed)
      rmse_all, report_all = mpr_all.report(num_simulations=1, trim_rate=0.0)
      self.assertEqual(rmse_exp, rmse_all)
      pd.util.testing.assert_frame_equal(report_exp, report_all)

if __name__ == "__main__":
  unittest.main()
