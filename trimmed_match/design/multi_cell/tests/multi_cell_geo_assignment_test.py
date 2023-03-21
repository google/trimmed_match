"""Tests for multi cell geo assignment generation."""
import pandas as pd
from trimmed_match.design.multi_cell import geo_blocking
from trimmed_match.design.multi_cell import multi_cell_geo_assignment

import unittest


class MultiCellGeoAssignmentTest(unittest.TestCase):

  def setUp(self):
    super(MultiCellGeoAssignmentTest, self).setUp()
    self.geo_level_data = pd.DataFrame({
        'geo': range(12),
        'response': [i * i for i in range(12)],
        'cost': [1, 2, 30, 4, 5, 60, 7, 8, 9, 10, 22, 3],
    })
    self.geo_blocks = geo_blocking.create_geo_blocks(
        self.geo_level_data, block_size=3)

  def testAssignmentNotEnoughCells(self):
    with self.assertRaisesRegex(
        ValueError, r'The number of cells must be positive, but got 0'):
      multi_cell_geo_assignment.generate_random_blocked_assignment(
          self.geo_blocks, [])

  def testAssignmentDifferentNumberOfCellsInGeoBlocksAndCellIds(self):
    with self.assertRaisesRegex(
        ValueError,
        r'The number of cell_ids and the size of the blocks should be the same.'
    ):
      multi_cell_geo_assignment.generate_random_blocked_assignment(
          self.geo_blocks, [1, 2])

  def testBlockAssignment(self):
    assignment = multi_cell_geo_assignment.generate_random_blocked_assignment(
        self.geo_blocks, [0, 1, 2], 0)
    expected = pd.DataFrame({
        'geo': range(12),
        'block': sorted([1, 2, 3, 4] * 3),
        'assignment': [2, 0, 1, 2, 1, 0, 2, 0, 1, 1, 2, 0],
    })
    pd.testing.assert_frame_equal(assignment, expected)

  def testBlockAssignmentGeoBlockNotSortedByBlock(self):
    assignment = multi_cell_geo_assignment.generate_random_blocked_assignment(
        pd.DataFrame({
            'geo': range(12),
            'block': [1, 2, 3, 4, 4, 3, 2, 1, 1, 2, 3, 4],
        }), [0, 1, 2], 0)
    expected = pd.DataFrame({
        'geo': [0, 7, 8, 1, 6, 9, 2, 5, 10, 3, 4, 11],
        'block': sorted([1, 2, 3, 4] * 3),
        'assignment': [2, 0, 1, 2, 1, 0, 2, 0, 1, 1, 2, 0],
    })
    pd.testing.assert_frame_equal(assignment, expected)

  def testBlockAssignmentCustomCellIds(self):
    assignment = multi_cell_geo_assignment.generate_random_blocked_assignment(
        self.geo_blocks, [1, 2, 3], 0)
    expected = pd.DataFrame({
        'geo': range(12),
        'block': sorted([1, 2, 3, 4] * 3),
        'assignment': [3, 1, 2, 3, 2, 1, 3, 1, 2, 2, 3, 1],
    })
    pd.testing.assert_frame_equal(assignment, expected)


if __name__ == '__main__':
  unittest.main()
