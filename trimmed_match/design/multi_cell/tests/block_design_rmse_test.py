import pandas as pd
from trimmed_match.design import common_classes
from trimmed_match.design.multi_cell import block_design_rmse
from trimmed_match.design.multi_cell import multi_cell_util

import unittest

_PRE_EXPERIMENT = common_classes.ExperimentPeriod.PRE_EXPERIMENT
_EXPERIMENT = common_classes.ExperimentPeriod.EXPERIMENT


class BlockDesignRMSE(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._geox_eval_data = pd.DataFrame({
        'block': [0, 0, 0],
        'geo': [1, 2, 3],
        'response': [10.0, 20.0, 30.0],
        'cost': [10.0, 20.0, 30.0],
    })
    self.cells = multi_cell_util.Cells.from_type(
        experiment_type=multi_cell_util.GeoXType.HEAVY_UP,
        num_cells=3,
        budgets=1.0,
    )

  def testMissingMandatoryColumn(self):
    with self.assertRaisesRegex(
        KeyError, r'geo_blocks_eval_data is missing {\'response\'}.'
    ):
      block_design_rmse.BlockDesignRMSE(
          self._geox_eval_data.drop(columns='response'), self.cells
      )

  def testDuplicatedGeo(self):
    with self.assertRaisesRegex(
        ValueError, r'Duplicated geo values in geo_blocks_eval_data.'
    ):
      block_design_rmse.BlockDesignRMSE(
          self._geox_eval_data.append(
              pd.DataFrame(
                  {'geo': [1], 'block': [0], 'response': [1], 'cost': [0.0]}
              )
          ),
          self.cells,
      )

  def testDuplicatedGeoPeriod(self):
    with self.assertRaisesRegex(
        ValueError, r'Duplicated geo-period values in geo_blocks_eval_data.'
    ):
      block_design_rmse.BlockDesignRMSE(
          pd.DataFrame({
              'geo': [1, 2, 3, 3],
              'period': [_EXPERIMENT, _EXPERIMENT, _EXPERIMENT, _EXPERIMENT],
              'block': [0, 0, 0, 0],
              'response': [10.0, 20.0, 30.0, 40.0],
              'cost': [10.0, 20.0, 30.0, 40.0],
          }),
          self.cells,
      )

    with self.assertRaisesRegex(
        ValueError,
        r'Duplicated date-geo-period values in geo_blocks_eval_data.',
    ):
      block_design_rmse.BlockDesignRMSE(
          pd.DataFrame({
              'geo': [1, 2, 2],
              'date': ['2022-01-01', '2022-01-02', '2022-01-02'],
              'period': [_PRE_EXPERIMENT, _EXPERIMENT, _EXPERIMENT],
              'block': [0, 0, 0],
              'response': [10.0, 20.0, 30.0],
              'cost': [10.0, 20.0, 30.0],
          }),
          self.cells,
      )

  def testMissingGeoPeriod(self):
    with self.assertRaisesRegex(
        ValueError, r'Missing geo-period values in geo_blocks_eval_data.'
    ):
      block_design_rmse.BlockDesignRMSE(
          pd.DataFrame({
              'geo': [1, 2, 1],
              'period': [_PRE_EXPERIMENT] * 2 + [_EXPERIMENT],
              'block': [0, 0, 0],
              'response': [10.0, 20.0, 30.0],
              'cost': [10.0, 20.0, 30.0],
          }),
          self.cells,
      )

  def testExperimentPeriodMissing(self):
    with self.assertRaisesRegex(
        ValueError, r'Missing experiment value 1 in geo_pairs_eval_data.period.'
    ):
      block_design_rmse.BlockDesignRMSE(
          pd.DataFrame({
              'period': [_PRE_EXPERIMENT] * 3,
              'block': [0, 0, 0],
              'geo': [1, 2, 3],
              'response': [10.0, 20.0, 30.0],
              'cost': [10.0, 20.0, 30.0],
          }),
          self.cells,
      )

  def testNegativeiROAS(self):
    with self.assertRaisesRegex(
        ValueError, r'iROAS must be positive, got -1.0.'
    ):
      block_design_rmse.BlockDesignRMSE(
          self._geox_eval_data,
          self.cells,
          hypothesized_iroas=-1.0,
      )

  def testDifferentNumberOfCells(self):
    with self.assertRaisesRegex(
        ValueError,
        r'geo_blocks_eval_data has 4 cells, while cells has 3 cells.',
    ):
      block_design_rmse.BlockDesignRMSE(
          pd.DataFrame({
              'block': [0, 0, 0, 0],
              'geo': [1, 2, 3, 4],
              'response': [10.0, 20.0, 30.0, 40.0],
              'cost': [10.0, 20.0, 30.0, 40.0],
          }),
          self.cells,
      )

  def testCorrectInitialization(self):
    block_design = block_design_rmse.BlockDesignRMSE(
        self._geox_eval_data,
        self.cells,
    )
    self.assertEqual(block_design.hypothesized_iroas, 0.0)
    self.assertEqual(block_design.block_size, 3)
    self.assertTrue(
        block_design.geo_blocks_eval_data.equals(self._geox_eval_data)
    )
    self.assertTrue(block_design.cells.equals(self.cells))
    self.assertEqual(block_design.num_blocks, 1)
    self.assertEqual(block_design.base_seed, 0)

  def testCorrectInitializationNonDefaultParams(self):
    block_design = block_design_rmse.BlockDesignRMSE(
        self._geox_eval_data,
        self.cells,
        hypothesized_iroas=1.0,
        base_seed=10,
    )
    self.assertEqual(block_design.hypothesized_iroas, 1.0)
    self.assertEqual(block_design.block_size, 3)
    self.assertTrue(
        block_design.geo_blocks_eval_data.equals(self._geox_eval_data)
    )
    self.assertTrue(block_design.cells.equals(self.cells))
    self.assertEqual(block_design.num_blocks, 1)
    self.assertEqual(block_design.base_seed, 10)

if __name__ == '__main__':
  unittest.main()
