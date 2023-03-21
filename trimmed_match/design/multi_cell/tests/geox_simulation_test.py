"""Tests for geox simulation."""

import numpy as np
import pandas as pd
from trimmed_match.design import common_classes
from trimmed_match.design.multi_cell import geox_simulation
from trimmed_match.design.multi_cell import multi_cell_util

import unittest
from absl.testing import parameterized


_PRE_EXPERIMENT = common_classes.ExperimentPeriod.PRE_EXPERIMENT
_EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT

GeoXType = common_classes.GeoXType
Cells = multi_cell_util.Cells


class TreatedCostResponseSimulationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.geox_eval_data = pd.DataFrame({
        "period": [_EXPERIMENT] * 2,
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "cost": [10.0, 20.0],
        "assignment": [0, 1],
    })
    self.iroas = {"control": 1.0, "treatment": 1.0}

  @parameterized.named_parameters(
      ("sutva", True, [0.0, 0.0]),
      ("not_sutva", False, [0.0, 0.0]),
  )
  def testGoDark(self, sutva, expected):
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_types=[GeoXType.CONTROL, GeoXType.GO_DARK],
        budgets=[0, None],
        cell_ids=[0, 1],
    )
    results = geox_simulation.simulate_multi_cell_geox_experiment(
        self.geox_eval_data, multi_cell, self.iroas, sutva
    )
    treated_results = results.loc[
        results.assignment == 1, ["response", "cost"]
    ].values.tolist()[0]
    self.assertListEqual(treated_results, expected)

  @parameterized.named_parameters(
      ("sutva", True, [60.0, 60.0]),
      ("not_sutva", False, [50.0, 50.0]),
  )
  def testHeavyUp(self, sutva, expected):
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_types=[GeoXType.CONTROL, GeoXType.HEAVY_UP],
        budgets=[0, 30],
        cell_ids=[0, 1],
    )
    results = geox_simulation.simulate_multi_cell_geox_experiment(
        self.geox_eval_data, multi_cell, self.iroas, sutva
    )
    treated_results = results.loc[
        results.assignment == 1, ["response", "cost"]
    ].values.tolist()[0]
    self.assertListEqual(treated_results, expected)

  @parameterized.named_parameters(
      ("sutva", True, [60.0, 40.0]),
      ("not_sutva", False, [50.0, 30.0]),
  )
  def testHoldBack(self, sutva, expected):
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_types=[GeoXType.CONTROL, GeoXType.HOLD_BACK],
        budgets=[None, 30],
        cell_ids=[0, 1],
    )
    results = geox_simulation.simulate_multi_cell_geox_experiment(
        self.geox_eval_data, multi_cell, self.iroas, sutva
    )
    treated_results = results.loc[
        results.assignment == 1, ["response", "cost"]
    ].values.tolist()[0]
    self.assertListEqual(treated_results, expected)


class ThreeCellsSimulationTest(unittest.TestCase):

  def test_simulate_multi_cell_geox_experiment_not_sutva(self):
    geo_pairs_eval_data = pd.DataFrame({
        "period": [_PRE_EXPERIMENT] * 6 + [_EXPERIMENT] * 6,
        "geo": [1, 2, 3, 4, 5, 6] * 2,
        "assignment": [1, 2, 3, 1, 2, 3] * 2,
        "response": [10, 20, 30, 40, 50, 60] * 2,
        "cost": [0] * 6 + [10, 20, 30, 40, 50, 60],
        "cost_channel0": [0] * 6 + [2, 4, 6, 8, 10, 12],
        "cost_channel1": [0] * 6 + [4, 8, 12, 16, 20, 24],
    })
    multi_cell = Cells.from_lists(
        cell_names=["control", "channel0_dark", "channel1_up"],
        cell_ids=[1, 2, 3],
        cell_types=[GeoXType.CONTROL, GeoXType.GO_DARK, GeoXType.HEAVY_UP],
        budgets=[0, None, 30],
        cost_columns=["cost", "cost_channel0", "cost_channel1"],
    )
    test_iroas = {"control": 3.0, "channel0_dark": 2.0, "channel1_up": 4.0}
    simulated_data = geox_simulation.simulate_multi_cell_geox_experiment(
        geox_data=geo_pairs_eval_data,
        multi_cell=multi_cell,
        iroas=test_iroas,
        adjust_budget_to_satisfy_sutva=False,
    )
    expected_data = pd.DataFrame({
        "period": [_PRE_EXPERIMENT] * 6 + [_EXPERIMENT] * 6,
        "geo": [1, 2, 3, 4, 5, 6] * 2,
        "assignment": [1, 2, 3, 1, 2, 3] * 2,
        "response": [10, 20, 30, 40, 50, 60] + [10, 12, 70, 40, 30, 140],
        "cost": [0] * 6 + [10, 16, 40, 40, 40, 80],
        "cost_channel0": [0] * 6 + [2, 0, 6, 8, 0, 12],
        "cost_channel1": [0] * 6 + [4, 8, 22, 16, 20, 44],
    })
    self.assertTrue(np.allclose(simulated_data, expected_data, rtol=1e-5))


class SimulationErrors(unittest.TestCase):

  def test_missing_iroas(self):
    geox_data = pd.DataFrame({
        "period": [_EXPERIMENT] * 2,
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "cost": [10.0, 20.0],
        "assignment": [0, 1],
    })
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_ids=[0, 1],
        cell_types=[GeoXType.CONTROL, GeoXType.HEAVY_UP],
        budgets=[0, 30],
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "An iroas is missing for cell(s) {'control'}."
    ):
      geox_simulation.simulate_multi_cell_geox_experiment(
          geox_data, multi_cell, iroas={"treatment": 1.0}
      )

  def testUnknownGeoXType(self):
    geox_data = pd.DataFrame({
        "period": [_EXPERIMENT] * 2,
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "cost": [10.0, 20.0],
        "assignment": [0, 1],
    })
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_types=[GeoXType.CONTROL, "UNKNOWN"],
        budgets=[None, 30],
        cell_ids=[0, 1],
    )
    with self.assertRaisesRegex(ValueError, "GeoxType UNKNOWN not supported."):
      geox_simulation.simulate_multi_cell_geox_experiment(
          geox_data, multi_cell, iroas={"control": None, "treatment": 1.0}
      )

  def test_missing_cost_column(self):
    geox_data = pd.DataFrame({
        "period": [_EXPERIMENT] * 2,
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "cost": [10.0, 20.0],
        "assignment": [0, 1],
    })
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_ids=[0, 1],
        cell_types=[GeoXType.CONTROL, GeoXType.HEAVY_UP],
        budgets=[0, 30],
        cost_columns=["cost", "cost_channel0"],
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError, "geox_data is missing column(s) {'cost_channel0'}."
    ):
      geox_simulation._checks_data_match_cell_config(geox_data, multi_cell)

  def test_go_dark_cell_with_non_zero_budget(self):
    geox_data = pd.DataFrame({
        "period": [_EXPERIMENT] * 2,
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "cost": [10.0, 20.0],
        "assignment": [0, 1],
    })
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_ids=[1, 2],
        cell_types=[GeoXType.CONTROL, GeoXType.GO_DARK],
        budgets=[0, 30],
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "The budget must be None or 0 for GO_DARK cells, but found"
            " {'budget': {'treatment': 30}}."
        ),
    ):
      geox_simulation._checks_data_match_cell_config(geox_data, multi_cell)

  def test_assignment_values_match_cell_ids(self):
    geox_data = pd.DataFrame({
        "period": [_EXPERIMENT] * 2,
        "pair": [0, 0],
        "geo": [1, 2],
        "response": [10.0, 20.0],
        "cost": [10.0, 20.0],
        "assignment": [0, 1],
    })
    multi_cell = Cells.from_lists(
        cell_names=["control", "treatment"],
        cell_ids=[1, 2],
        cell_types=[GeoXType.CONTROL, GeoXType.HEAVY_UP],
        budgets=[0, 30],
    )
    with self.assertRaisesWithLiteralMatch(
        ValueError,
        (
            "geox_data assignment values do not match cell_id values in"
            " multi-cell."
        ),
    ):
      geox_simulation._checks_data_match_cell_config(geox_data, multi_cell)


if __name__ == "__main__":
  unittest.main()
