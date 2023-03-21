"""Tests for geo blocking generation."""
import numpy as np
import pandas as pd
from trimmed_match.design.multi_cell import geo_blocking

import unittest


class GeoBlockingTest(unittest.TestCase):

  def setUp(self):
    super(GeoBlockingTest, self).setUp()
    self.geo_level_data = pd.DataFrame({
        'geo': range(9),
        'response': [i * i for i in range(9)],
        'cost': [1, 2, 30, 4, 5, 60, 7, 8, 9],
    })

  def testGeoBlockingMissingColumn(self):
    with self.assertRaisesRegex(ValueError,
                                r'geo_level_data is missing {\'response\'}'):
      geo_blocking.create_geo_blocks(
          self.geo_level_data.drop(columns=['response']), block_size=3)

  def testGeoBlockingMultipleObservationsPerGeo(self):
    with self.assertRaisesRegex(ValueError,
                                r'geo_level_data has duplicated geos'):
      geo_blocking.create_geo_blocks(
          self.geo_level_data.append(
              pd.DataFrame({
                  'geo': 1,
                  'response': 3.5,
                  'cost': 1.0
              }, index=[0])),
          block_size=3)

  def testSamePairsAsStandardTrimmedMatch(self):
    test_data = pd.DataFrame({
        'date':
            pd.to_datetime([
                '2019-01-01', '2019-01-01', '2019-01-01', '2019-01-01'
            ]),
        'geo': [1, 2, 3, 4],
        'response': [3, 7, 3, 7],
        'cost': [2.5, 4.5, 2.5, 11],
    })
    pairs = geo_blocking.create_geo_blocks(test_data, block_size=2)
    expected_pairs = pd.DataFrame({'geo': [1, 3, 2, 4], 'block': [1, 1, 2, 2]})
    self.assertTrue(pairs.equals(expected_pairs))

  def testGeoBlocking(self):
    result_three_blocks = geo_blocking.create_geo_blocks(
        self.geo_level_data, block_size=3)
    result_four_blocks = geo_blocking.create_geo_blocks(
        self.geo_level_data, block_size=4)
    expected_three_blocks = pd.DataFrame({
        'geo': range(9),
        'block': np.repeat([1, 2, 3], 3)
    })
    expected_four_blocks = pd.DataFrame({
        'geo': range(8),
        'block': np.repeat([1, 2], 4)
    })
    self.assertTrue(result_three_blocks[['geo', 'block'
                                        ]].equals(expected_three_blocks))
    self.assertTrue(result_four_blocks[['geo',
                                        'block']].equals(expected_four_blocks))
    self.assertTrue(result_four_blocks[['geo',
                                        'block']].equals(expected_four_blocks))

  def testGeoBlockingNewWeights(self):
    result = geo_blocking.create_geo_blocks(
        self.geo_level_data, block_size=3, matching_metrics={
            'response': 1,
            'cost': 10
        })
    expected = pd.DataFrame({
        'geo': [0, 1, 3, 4, 6, 7, 2, 5, 8],
        'block': np.repeat([1, 2, 3], 3)
    })
    self.assertTrue(result[['geo', 'block']].equals(expected))

  def testGeoBlockingCustomMetrics(self):
    df = pd.DataFrame({
        'geo': range(10),
        'metric': [i * i for i in range(10)],
    })
    result = geo_blocking.create_geo_blocks(
        df, block_size=5, matching_metrics={'metric': 1})
    expected = pd.DataFrame({
        'geo': range(10),
        'block': np.repeat([1, 2], 5)
    })
    self.assertTrue(result[['geo', 'block']].equals(expected))

if __name__ == '__main__':
  unittest.main()
