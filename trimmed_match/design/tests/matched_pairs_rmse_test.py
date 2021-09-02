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
import pandas as pd

from trimmed_match.design import common_classes
from trimmed_match.design import matched_pairs_rmse
import unittest

GeoLevelData = common_classes.GeoLevelData
GeoXType = common_classes.GeoXType
GeoLevelPotentialOutcomes = common_classes.GeoLevelPotentialOutcomes
MatchedPairsRMSE = matched_pairs_rmse.MatchedPairsRMSE


class ConstructPotentialOutcomesTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._geox_eval_data = pd.DataFrame({
        "geo": [1, 2],
        "response": [10, 20],
        "spend": [10, 20]
    })
    self._budget = 30
    self._hypothesized_iroas = 1

  def testGoDark(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.GO_DARK, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = {
        1:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=1, response=10, spend=20),
                treated=GeoLevelData(geo=1, response=0, spend=0)),
        2:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=2, response=20, spend=40),
                treated=GeoLevelData(geo=2, response=0, spend=0))
    }
    self.assertDictEqual(expected, potential_outcomes)

  def testGoDarkWithHeavyUpControl(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = {
        1:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=1, response=20, spend=20),
                treated=GeoLevelData(geo=1, response=0, spend=0)),
        2:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=2, response=40, spend=40),
                treated=GeoLevelData(geo=2, response=0, spend=0))
    }
    self.assertDictEqual(expected, potential_outcomes)

  def testHeavyUp(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.HEAVY_UP, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = {
        1:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=1, response=10, spend=10),
                treated=GeoLevelData(geo=1, response=30.0, spend=30.0)),
        2:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=2, response=20, spend=20),
                treated=GeoLevelData(geo=2, response=60.0, spend=60.0))
    }
    self.assertDictEqual(expected, potential_outcomes)

  def testHeavyDown(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.HEAVY_DOWN, self._geox_eval_data, self._budget,
        self._hypothesized_iroas)
    expected = {
        1:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=1, response=10, spend=10),
                treated=GeoLevelData(geo=1, response=0.0, spend=0.0)),
        2:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=2, response=20, spend=20),
                treated=GeoLevelData(geo=2, response=0.0, spend=0.0))
    }
    self.assertDictEqual(expected, potential_outcomes)

  def testHoldBack(self):
    potential_outcomes = matched_pairs_rmse._construct_potential_outcomes(
        GeoXType.HOLD_BACK, self._geox_eval_data,
        (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
        self._hypothesized_iroas)
    expected = {
        1:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=1, response=10, spend=0.0),
                treated=GeoLevelData(geo=1, response=30.0, spend=20.0)),
        2:
            GeoLevelPotentialOutcomes(
                controlled=GeoLevelData(geo=2, response=20, spend=0.0),
                treated=GeoLevelData(geo=2, response=60.0, spend=40.0))
    }
    self.assertDictEqual(expected, potential_outcomes)

  def testUnknownGeoXType(self):
    """Checks an error is raised if the GeoX type is unknown."""
    with self.assertRaisesRegex(ValueError, "Unknown geox_type: \'UNKNOWN\'"):
      matched_pairs_rmse._construct_potential_outcomes(
          "UNKNOWN", self._geox_eval_data,
          (self._budget * 2.0 / self._geox_eval_data.spend.sum()),
          self._hypothesized_iroas)


class IsPairedTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._ordered_list_one = [1, 1, 2, 2, 3, 3]
    self._ordered_list_two = [1, 2, 2, 2, 3, 3]

  def testIsPaired(self):
    self.assertTrue(matched_pairs_rmse._is_paired(self._ordered_list_one))

  def testIsNotPaired(self):
    self.assertFalse(matched_pairs_rmse._is_paired(self._ordered_list_two))


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
    self._budget = 10
    self._hypothesized_iroas = 1

    # TODO(b/147698415): adding a more complex test example here

  def AssertEqualGeoLevelData(self, outcome1: GeoLevelData,
                              outcome2: GeoLevelData):
    """Checks whether two GeoLevelDatas are equal."""
    self.assertEqual(outcome1.geo, outcome2.geo)
    self.assertEqual(outcome1.response, outcome2.response)
    self.assertEqual(outcome1.spend, outcome2.spend)

  def testHypothesizedIroasNegative(self):
    """Checks an error is raised if the hypothesized iROAS is negative."""
    with self.assertRaisesRegex(ValueError, "iROAS must be positive, got -1.0"):
      MatchedPairsRMSE(GeoXType.GO_DARK, self._geo_pairs_eval_data,
                       self._budget, -1.0)

  def testGeosNotUnique(self):
    """Checks an error is raised if geos are duplicated."""
    geo_pairs_eval_data = self._geo_pairs_eval_data.copy()
    geo_pairs_eval_data.loc[geo_pairs_eval_data["geo"] == 3, "geo"] = 1
    with self.assertRaisesRegex(ValueError,
                                "Geos are not unique in geo_pairs_eval_data"):
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

  def testSimulateGeoXDataRandomization(self):
    """Checks randomization within the pair."""
    mpr = MatchedPairsRMSE(GeoXType.GO_DARK, self._geo_pairs_eval_data,
                           self._budget, self._hypothesized_iroas)
    geox_data = mpr._simulate_geox_data(0)
    for pair, value in geox_data.items():
      expected = mpr._paired_geos[pair].values()
      self.assertSetEqual(
          set(expected), set([value.controlled.geo, value.treated.geo]))

  def testSimulatedGeoXDataValue(self):
    """Checks the data accuracy."""
    for geox_type in GeoXType:
      if geox_type == GeoXType.CONTROL:
        continue
      mpr = MatchedPairsRMSE(geox_type, self._geo_pairs_eval_data, self._budget,
                             self._hypothesized_iroas)
      geox_data = mpr._simulate_geox_data(0)
      for _, value in geox_data.items():
        treatment_geo = value.treated.geo
        control_geo = value.controlled.geo
        treatment_geo_outcome = mpr._potential_outcomes[treatment_geo].treated
        control_geo_outcome = mpr._potential_outcomes[control_geo].controlled
        self.AssertEqualGeoLevelData(treatment_geo_outcome, value.treated)
        self.AssertEqualGeoLevelData(control_geo_outcome, value.controlled)

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
    dataframe = pd.DataFrame({
        "geo": [1, 2, 3, 4, 5, 6, 7, 8],
        "response": [10, 11, 20, 30, 30, 33, 40, 48],
        "spend": [1.0, 2.0, 3.0, 7.0, 3.0, 5.0, 4.0, 9.0],
        "pair": [1, 1, 2, 2, 3, 3, 4, 4],
    })
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


if __name__ == "__main__":
  unittest.main()
