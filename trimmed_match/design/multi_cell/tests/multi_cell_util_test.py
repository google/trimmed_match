from absl.testing import parameterized
import numpy as np
import pandas as pd
from trimmed_match.design.multi_cell import multi_cell_util

import unittest

GeoXType = multi_cell_util.GeoXType


class InferBlockSizeTest(unittest.TestCase):

  def testInferBlockSize(self):
    test_list = [1, 2, 3, 1, 2, 3]
    test_series = pd.Series(test_list)
    test_array = np.array(test_list)
    self.assertEqual(multi_cell_util.infer_block_size(test_list), 2)
    self.assertEqual(multi_cell_util.infer_block_size(test_series), 2)
    self.assertEqual(multi_cell_util.infer_block_size(test_array), 2)

  def testInferBlockSizeLessThanTwoElements(self):
    with self.assertRaisesRegex(ValueError,
                                r'The input length must be greater than 1, ' +
                                'got 1.'):
      multi_cell_util.infer_block_size([1])

  def testInferBlockSizeIrregularBlocks(self):
    with self.assertRaisesRegex(ValueError,
                                r'The number of geos per block is not ' +
                                'constant.'):
      multi_cell_util.infer_block_size([1, 2, 3, 2, 3])

  def testInferBlockSizeLessThanTwoGeosPerBlock(self):
    with self.assertRaisesRegex(ValueError,
                                r'The number of geos per block ' +
                                'must be >= 2, got 1.'):
      multi_cell_util.infer_block_size([1, 2, 3])


class CellsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.cell_names = ['Control', 'Channel_1', 'Channel_2']
    self.cell_ids = [10, 15, 20]
    self.cell_types = [GeoXType.CONTROL, GeoXType.HEAVY_UP, GeoXType.GO_DARK]
    self.budgets = [0, 10, 20]
    self.cost_columns = ['cost', 'cost_channel_1', 'cost_channel_2']
    self.cells = pd.DataFrame({'cell_name': self.cell_names,
                               'cell_type': self.cell_types,
                               'cell_id': self.cell_ids,
                               'budget': self.budgets,
                               'cost_column': self.cost_columns})

  def testCorrectInitializationFromDataFrame(self):
    self.assertTrue(multi_cell_util.Cells(self.cells).equals(self.cells))

  def testMissingColumn(self):
    with self.assertRaisesRegex(
        ValueError, r'The dataframe must contain the columns ' +
        'cell_name, cell_type, cell_id, budget.'):
      multi_cell_util.Cells(self.cells.drop(columns=['cell_id']))

  def testMixHoldBackWithOtherTypes(self):
    with self.assertRaisesRegex(
        ValueError, r'Mixing HOLD_BACK with HEAVY-UP/GO-DARK is not allowed.'):
      multi_cell_util.Cells(
          pd.DataFrame({
              'cell_name': self.cell_names,
              'cell_type': [
                  GeoXType.CONTROL, GeoXType.HOLD_BACK, GeoXType.GO_DARK
              ],
              'cell_id': self.cell_ids,
              'budget': self.budgets,
              'cost_column': self.cost_columns,
          }))

    with self.assertRaisesRegex(
        ValueError, r'Mixing HOLD_BACK with HEAVY-UP/GO-DARK is not allowed.'):
      multi_cell_util.Cells(
          pd.DataFrame({
              'cell_name': self.cell_names,
              'cell_type': [
                  GeoXType.CONTROL, GeoXType.HOLD_BACK, GeoXType.HEAVY_UP
              ],
              'cell_id': self.cell_ids,
              'budget': self.budgets,
              'cost_column': self.cost_columns,
          }))

  def testMixHoldBackWithControl(self):
    cells = multi_cell_util.Cells(
        pd.DataFrame({
            'cell_name': self.cell_names,
            'cell_type': [
                GeoXType.CONTROL, GeoXType.HOLD_BACK, GeoXType.HOLD_BACK
            ],
            'cell_id': self.cell_ids,
            'budget': self.budgets,
            'cost_column': self.cost_columns,
        }))
    self.assertTrue(
        cells.equals(
            pd.DataFrame({
                'cell_name': self.cell_names,
                'cell_type': [
                    GeoXType.CONTROL, GeoXType.HOLD_BACK, GeoXType.HOLD_BACK
                ],
                'cell_id': self.cell_ids,
                'budget': self.budgets,
                'cost_column': self.cost_columns,
            })))

  def testMixGoDarkNotBAUWithOtherTypes(self):
    with self.assertRaisesRegex(
        ValueError, r'GO_DARK_TREATMENT_NOT_BAU_CONTROL is not supported '
        'for multi cell experiments.'):
      multi_cell_util.Cells(
          pd.DataFrame({
              'cell_name': self.cell_names,
              'cell_type': [
                  GeoXType.CONTROL, GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL,
                  GeoXType.GO_DARK
              ],
              'cell_id': self.cell_ids,
              'budget': self.budgets,
              'cost_column': self.cost_columns,
          }))

  def testGoDarkNotBAUForTwoCellsExperiment(self):
    cells = multi_cell_util.Cells(
        pd.DataFrame({
            'cell_name': ['cell_1', 'cell_2'],
            'cell_type': [
                GeoXType.CONTROL,
                GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL,
            ],
            'cell_id': [1, 2],
            'budget': [0.0, 20.0],
            'cost_column': ['cost', 'cost'],
        }))
    self.assertTrue(
        cells.equals(
            pd.DataFrame({
                'cell_name': ['cell_1', 'cell_2'],
                'cell_type': [
                    GeoXType.CONTROL,
                    GeoXType.GO_DARK_TREATMENT_NOT_BAU_CONTROL,
                ],
                'cell_id': [1, 2],
                'budget': [0.0, 20.0],
                'cost_column': ['cost', 'cost'],
            })))

  def testNoControlGroupError(self):
    with self.assertRaisesRegex(
        ValueError, r'One control cell is required.'):
      multi_cell_util.Cells(
          pd.DataFrame({
              'cell_name': ['cell_1', 'cell_2'],
              'cell_type': [
                  GeoXType.GO_DARK, GeoXType.HEAVY_UP,
              ],
              'cell_id': [1, 2],
              'budget': [0.0, 20.0],
              'cost_column': ['cost', 'cost'],
          }))

  def testNoGroupWithNoBudgetChange(self):
    with self.assertRaisesRegex(
        ValueError, r'One group with no spend change is required.'):
      multi_cell_util.Cells(
          pd.DataFrame({
              'cell_name': ['cell_1', 'cell_2', 'cell_3'],
              'cell_type': [
                  GeoXType.CONTROL, GeoXType.HEAVY_UP, GeoXType.GO_DARK,
              ],
              'cell_id': [1, 2, 3],
              'budget': [10.0, 20.0, 30.0],
              'cost_column': ['cost', 'cost', 'cost'],
          }))

  def testCorrectInitializationFromList(self):
    self.assertTrue(
        multi_cell_util.Cells.from_lists(
            cell_names=self.cell_names,
            cell_types=self.cell_types,
            cell_ids=self.cell_ids,
            budgets=self.budgets,
            cost_columns=self.cost_columns).equals(self.cells))

  def testCorrectInitializationFromListOptional(self):
    self.assertTrue(
        multi_cell_util.Cells.from_lists(
            cell_types=self.cell_types,
            budgets=self.budgets,
        ).equals(
            pd.DataFrame({
                'cell_name': ['Cell_0', 'Cell_1', 'Cell_2'],
                'cell_type': self.cell_types,
                'cell_id': [0, 1, 2],
                'budget': self.budgets,
                'cost_column': ['cost'] * 3,
            })))

  def testFailInitializationFromListIncorrectCellNames(self):
    with self.assertRaisesRegex(
        ValueError,
        r'cell_names and cell_types should have the same length'):
      multi_cell_util.Cells.from_lists(
          cell_names=self.cell_names[:-1],
          cell_types=self.cell_types,
          cell_ids=self.cell_ids,
          budgets=self.budgets)

  def testFailInitializationFromListIncorrectCellIds(self):
    with self.assertRaisesRegex(
        ValueError,
        r'cell_ids and cell_types should have the same length'):
      multi_cell_util.Cells.from_lists(
          cell_names=self.cell_names,
          cell_types=self.cell_types,
          cell_ids=self.cell_ids[:-1],
          budgets=self.budgets)

  def testFailInitializationFromListIncorrectBudgets(self):
    with self.assertRaisesRegex(
        ValueError,
        r'budgets and cell_types should have the same length'):
      multi_cell_util.Cells.from_lists(
          cell_names=self.cell_names,
          cell_types=self.cell_types,
          cell_ids=self.cell_ids,
          budgets=self.budgets[:-1])

  def testFailInitializationFromListIncorrectCostColumns(self):
    with self.assertRaisesRegex(
        ValueError,
        r'cost_columns and cell_types should have the same length'):
      multi_cell_util.Cells.from_lists(
          cell_names=self.cell_names,
          cell_types=self.cell_types,
          cell_ids=self.cell_ids,
          budgets=self.budgets,
          cost_columns=self.cost_columns[:-1])

  def testCorrectInitializationFromTypeOptional(self):
    self.assertTrue(
        multi_cell_util.Cells.from_type(
            experiment_type=GeoXType.HEAVY_UP,
            num_cells=3,
            budgets=1.0
            ).equals(
                pd.DataFrame({
                    'cell_name': ['Cell_0', 'Cell_1', 'Cell_2'],
                    'cell_type': [
                        GeoXType.CONTROL, GeoXType.HEAVY_UP, GeoXType.HEAVY_UP
                    ],
                    'cell_id': [0, 1, 2],
                    'budget': [0.0, 1.0, 1.0],
                    'cost_column': ['cost'] * 3,
                })))

    self.assertTrue(
        multi_cell_util.Cells.from_type(
            experiment_type=GeoXType.HEAVY_UP,
            num_cells=3,
            budgets=[0, 2, 3],
            ).equals(
                pd.DataFrame({
                    'cell_name': ['Cell_0', 'Cell_1', 'Cell_2'],
                    'cell_type': [
                        GeoXType.CONTROL, GeoXType.HEAVY_UP, GeoXType.HEAVY_UP
                    ],
                    'cell_id': [0, 1, 2],
                    'budget': [0, 2, 3],
                    'cost_column': ['cost'] * 3,
                })))

  @parameterized.parameters(GeoXType.HEAVY_UP, GeoXType.GO_DARK)
  def testCorrectInitializationFromType(self, experiment_type):

    self.assertTrue(
        multi_cell_util.Cells.from_type(
            experiment_type=experiment_type,
            num_cells=3,
            budgets=self.budgets,
            cell_names=self.cell_names,
            cell_ids=self.cell_ids).equals(
                pd.DataFrame({
                    'cell_name': self.cell_names,
                    'cell_type': [
                        GeoXType.CONTROL, experiment_type, experiment_type
                    ],
                    'cell_id': self.cell_ids,
                    'budget': self.budgets,
                    'cost_column': ['cost'] * 3,
                })))

  def testFailInitializationFromTypeIncorrectCellNames(self):
    with self.assertRaisesRegex(
        ValueError,
        r'the length of cell_names is 2, but num_cells is 3'):
      multi_cell_util.Cells.from_type(
          experiment_type=GeoXType.HEAVY_UP,
          num_cells=3,
          budgets=self.budgets,
          cell_names=self.cell_names[:-1],
          cell_ids=self.cell_ids)

  def testFailInitializationFromTypeIncorrectCellIds(self):
    with self.assertRaisesRegex(
        ValueError,
        r'the length of cell_ids is 2, but num_cells is 3'):
      multi_cell_util.Cells.from_type(
          experiment_type=GeoXType.HEAVY_UP,
          num_cells=3,
          cell_names=self.cell_names,
          cell_ids=self.cell_ids[:-1],
          budgets=self.budgets)

  def testFailInitializationFromTypeIncorrectBudgets(self):
    with self.assertRaisesRegex(
        ValueError,
        r'the length of budget is 2, but num_cells is 3'):
      multi_cell_util.Cells.from_type(
          experiment_type=GeoXType.HEAVY_UP,
          num_cells=3,
          budgets=self.budgets[:-1],
          cell_names=self.cell_names,
          cell_ids=self.cell_ids)

  def testFailInitializationFromTypeIncorrectCostColumns(self):
    with self.assertRaisesRegex(
        ValueError,
        r'the length of cost_columns is 2, but num_cells is 3'):
      multi_cell_util.Cells.from_type(
          experiment_type=GeoXType.HEAVY_UP,
          num_cells=3,
          budgets=self.budgets,
          cell_names=self.cell_names,
          cell_ids=self.cell_ids,
          cost_columns=self.cost_columns[:-1])

if __name__ == '__main__':
  unittest.main()
