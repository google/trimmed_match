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

"""Tests for trimmed_match.design.tests.plot_utilities."""

from unittest import mock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trimmed_match.design import common_classes
from trimmed_match.design import plot_utilities
from trimmed_match.design import trimmed_match_design
import unittest

GeoXType = common_classes.GeoXType
TimeWindow = common_classes.TimeWindow
TrimmedMatchGeoXDesign = trimmed_match_design.TrimmedMatchGeoXDesign


class PlotUtilitiesTest(unittest.TestCase):

  def setUp(self):
    """This method will be run before each of the test methods in the class."""
    super(PlotUtilitiesTest, self).setUp()
    self.response = 'response'
    self.num_pairs = 2
    self.results = pd.DataFrame({
        'num_pairs_filtered': [0, 1],
        'experiment_response': [10, 20],
        'budget': [10, 10],
        'iroas': [0, 0],
        'rmse': [2, 1],
        'rmse_cost_adjusted': [2, 3],
    })
    self.test_data = pd.DataFrame({
        'date':
            pd.to_datetime([
                '2019-01-01', '2019-10-01', '2019-01-01', '2019-10-01',
                '2019-01-01', '2019-10-01', '2019-01-01', '2019-10-01'
            ]),
        'geo': [1, 1, 2, 2, 3, 3, 4, 4],
        'response': [1, 2, 2, 5, 1, 2, 3, 4],
        'spend': [1, 1.5, 2, 2.5, 1, 1.5, 5, 6],
        'metric': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    })
    self.design_window = TimeWindow(
        pd.Timestamp('2019-01-01'), pd.Timestamp('2019-10-01'))
    self.evaluation_window = TimeWindow(
        pd.Timestamp('2019-09-01'), pd.Timestamp('2019-10-01'))
    self.test_class = TrimmedMatchGeoXDesign(
        GeoXType.HEAVY_UP,
        pretest_data=self.test_data,
        response='response',
        matching_metrics={'response': 1.0},
        time_window_for_design=self.design_window,
        time_window_for_eval=self.evaluation_window)
    _, geo_level_eval_data = self.test_class._create_geo_pairs(
        spend_proxy='spend', use_cross_validation=False)
    self.geo_level_eval_data = geo_level_eval_data
    self.geo_level_eval_data['assignment'] = [0, 1, 0, 1]

  def testHumanReadableFormat(self):
    numbers = [123, 10765, 13987482, 8927462746, 1020000000000]
    numb_formatted = [
        plot_utilities.human_readable_number(num) for num in numbers
    ]
    self.assertEqual(numb_formatted, ['123', '10.8K', '14M', '8.93B', '1.02tn'])

  def testPlotDesign(self):
    with mock.patch.object(plot_utilities, 'plt') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      mock_plt.figure.assert_called_with(figsize=(20, 10))
      mock_plt.close.assert_called()

    with mock.patch('matplotlib.legend.Legend') as legend:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertListEqual(legend.call_args_list[0][0][2],
                           ['RMSE', 'Cost adjusted RMSE'])

    with mock.patch.object(plt.Axes, 'plot') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertTrue(pd.Series(self.results.rmse).equals(
          mock_plt.call_args_list[0][0][1]))
      self.assertTrue(pd.Series(self.results.num_pairs_filtered).equals(
          mock_plt.call_args_list[0][0][0]))
      self.assertTrue(pd.Series(self.results.rmse_cost_adjusted).equals(
          mock_plt.call_args_list[1][0][1]))
      self.assertTrue(pd.Series(self.results.num_pairs_filtered).equals(
          mock_plt.call_args_list[1][0][0]))
      mock_plt.assert_called()

    with mock.patch.object(plt.Axes, 'text') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertTupleEqual(mock_plt.call_args_list[0][0], (1, 2.05, '10'))
      self.assertTupleEqual(mock_plt.call_args_list[1][0], (2, 1.05, '20'))

    with mock.patch.object(plt.Axes, 'set_xlim') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertTupleEqual(mock_plt.call_args_list[3][0], (-1, 2))

    with mock.patch.object(plt.Axes, 'set_ylim') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertTupleEqual(mock_plt.call_args_list[3][0], (0.95, 3.05))

    with mock.patch.object(plt.Axes, 'set_title') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertEqual(mock_plt.call_args_list[0][
          0][0], 'RMSE of iROAS w.r.t. response (total pairs: 2)')

    with mock.patch.object(plt.Axes, 'set_xlabel') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertEqual(mock_plt.call_args_list[0][0][0],
                       '#(Excluded geo pairs)')

    with mock.patch.object(plt.Axes, 'set_ylabel') as mock_plt:
      plot_utilities.plot_candidate_design_rmse(self.response, self.num_pairs,
                                                self.results)
      self.assertEqual(mock_plt.call_args_list[0][0][0], 'RMSE')

  def testOutputChosenDesign(self):
    with mock.patch.object(plot_utilities, 'sns') as mock_plt:
      plot_utilities.output_chosen_design(self.test_data,
                                          self.geo_level_eval_data,
                                          self.response, 'spend', 0,
                                          self.evaluation_window)
      tmp_treat = self.geo_level_eval_data.loc[
          self.geo_level_eval_data['assignment'] == 1,
          'response'].apply(np.sqrt)
      tmp_cont = self.geo_level_eval_data.loc[
          self.geo_level_eval_data['assignment'] == 0,
          'response'].apply(np.sqrt)
      self.assertTrue(
          tmp_treat.equals(mock_plt.regplot.call_args_list[0][1]['x']))
      self.assertTrue(
          tmp_cont.equals(mock_plt.regplot.call_args_list[0][1]['y']))

    with mock.patch.object(plt.Axes, 'set_xlabel') as mock_plt:
      plot_utilities.output_chosen_design(self.test_data,
                                          self.geo_level_eval_data,
                                          self.response, 'spend', 0,
                                          self.evaluation_window)
      self.assertEqual(mock_plt.call_args_list[1][0][0], 'treatment')
      self.assertEqual(mock_plt.call_args_list[3][0][0], 'treatment')
      self.assertEqual(mock_plt.call_args_list[4][0][0], 'date')
      self.assertEqual(mock_plt.call_args_list[5][0][0], 'date')

    with mock.patch.object(plt.Axes, 'set_ylabel') as mock_plt:
      plot_utilities.output_chosen_design(self.test_data,
                                          self.geo_level_eval_data,
                                          self.response, 'spend', 0,
                                          self.evaluation_window)
      self.assertEqual(mock_plt.call_args_list[1][0][0], 'control')
      self.assertEqual(mock_plt.call_args_list[3][0][0], 'control')
      self.assertEqual(mock_plt.call_args_list[4][0][0], self.response)
      self.assertEqual(mock_plt.call_args_list[5][0][0], 'spend')

    with mock.patch.object(pd.DataFrame, 'plot') as mock_plt:
      plot_utilities.output_chosen_design(self.test_data,
                                          self.geo_level_eval_data,
                                          self.response, 'spend', 0,
                                          self.evaluation_window)
      obtained = {
          key: value
          for key, value in mock_plt.call_args_list[0][1].items()
          if key in ['x', 'y', 'color', 'label']
      }
      self.assertDictEqual(obtained, {
          'x': 'date',
          'y': self.response,
          'color': 'black',
          'label': 'treatment',
      })
      obtained = {
          key: value
          for key, value in mock_plt.call_args_list[1][1].items()
          if key in ['x', 'y', 'color', 'label']
      }
      self.assertDictEqual(obtained, {
          'x': 'date',
          'y': self.response,
          'color': 'red',
          'label': 'control',
      })
      obtained = {
          key: value
          for key, value in mock_plt.call_args_list[2][1].items()
          if key in ['x', 'y', 'color', 'label']
      }
      self.assertDictEqual(obtained, {
          'x': 'date',
          'y': 'spend',
          'color': 'black',
          'label': 'treatment',
      })
      obtained = {
          key: value
          for key, value in mock_plt.call_args_list[3][1].items()
          if key in ['x', 'y', 'color', 'label']
      }
      self.assertDictEqual(obtained, {
          'x': 'date',
          'y': 'spend',
          'color': 'red',
          'label': 'control',
      })

    with mock.patch('matplotlib.legend.Legend') as legend:
      plot_utilities.output_chosen_design(self.test_data,
                                          self.geo_level_eval_data,
                                          self.response, 'spend', 0,
                                          self.evaluation_window)
      self.assertListEqual(legend.call_args_list[1][0][2], [
          'treatment', 'control', 'evaluation window'
      ])
      self.assertListEqual(legend.call_args_list[3][0][2], [
          'treatment', 'control', 'evaluation window'
      ])

  def testPlotPairedComparison(self):
    with mock.patch.object(plot_utilities, 'sns') as mock_plt:
      plot_utilities.plot_paired_comparison(self.test_data,
                                            self.geo_level_eval_data,
                                            self.response, 0,
                                            self.design_window,
                                            self.evaluation_window)
      data_to_plot = pd.merge(
          self.test_data,
          self.geo_level_eval_data[['geo', 'pair', 'assignment']],
          how='left',
          on='geo')
      data_to_plot.sort_values(by='assignment')
      self.assertTrue(
          mock_plt.FacetGrid.call_args_list[0][0][0].equals(data_to_plot))
      args = ['col', 'hue', 'col_wrap', 'sharey', 'sharex', 'height', 'aspect']
      obtained = {
          key: value
          for key, value in mock_plt.FacetGrid.call_args_list[0][1].items()
          if key in args
      }
      self.assertDictEqual(
          obtained, {
              'col': 'pair',
              'hue': 'assignment',
              'col_wrap': 2,
              'sharey': False,
              'sharex': False,
              'height': 3,
              'aspect': 2,
          })


if __name__ == '__main__':
  unittest.main()
