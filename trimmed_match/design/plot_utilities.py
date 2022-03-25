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
"""Functions to plot the results and diagnostics of a TrimmedMatch design."""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from trimmed_match.design import common_classes
from trimmed_match.design import util

TimeWindow = common_classes.TimeWindow


def plot_candidate_design_rmse(
    response: str, num_pairs: int,
    results: pd.DataFrame) -> Dict[Tuple[float, float], plt.Axes]:
  """Plot the RMSE curve for a set of candidate designs.

  Args:
    response: str, primary response variable used for the design.
    num_pairs: int, total number of pairs in the design.
    results: pd.DataFrame, with columns (pair_index,
      experiment_response, experiment_spend, spend_response_ratio, budget,
      iroas, rmse, proportion_cost_in_experiment).

  Returns:
    axes_dict: a dictionary with keys (budget, iroas) with the plot of the
      RMSE values for the design with corresponding budget and iROAS as a
      function of the number of excluded pairs.
  """
  budget_list = results['budget'].unique()
  iroas_list = results['iroas'].unique()
  axes_dict = {}
  for budget in budget_list:
    for iroas in iroas_list:
      result = results[(results['budget'] == budget)
                       & (results['iroas'] == iroas)].reset_index(drop=True)
      hlines = range(
          0,
          int(max(max(result['rmse']), max(result['rmse_cost_adjusted']))) + 1)

      delta = (max(result['rmse']) - min(result['rmse'])) * 0.05

      fig = plt.figure(figsize=(20, 10))
      ax = fig.add_subplot(1, 1, 1)
      ax.plot(
          result['pair_index'], result['rmse'], 'blue', label='RMSE')
      ax.plot(
          result['pair_index'],
          result['rmse_cost_adjusted'],
          'red',
          label='Cost adjusted RMSE')
      ax.set_xlim(
          min(result['pair_index']) - 1,
          max(result['pair_index']) + 1)
      ax.set_ylim(
          min(result.rmse) - delta,
          max(result.rmse_cost_adjusted) + delta)
      ax.legend()
      ax.hlines(
          y=hlines,
          xmin=min(result['pair_index']),
          xmax=max(result['pair_index']),
          colors='gray',
          linestyles='dashed')
      for _, row in result.iterrows():
        ax.text(
            row.pair_index + 1, row.rmse + delta,
            '{}'.format(
                util.human_readable_number(row.experiment_response)))
      ax.set_xlabel('Pairing number')
      ax.set_ylabel('RMSE')
      ax.set_title(
          f'''RMSE of iROAS w.r.t. {response} (total pairs: {num_pairs})''')
      axes_dict[(budget, iroas)] = fig
      plt.close()

  return axes_dict


def output_chosen_design(
    pretest_data: pd.DataFrame,
    geo_level_eval_data: pd.DataFrame,
    response: str,
    spend_proxy: str,
    time_window_for_eval: TimeWindow,
    group_control: int = common_classes.GeoAssignment.CONTROL,
    group_treatment: int = common_classes.GeoAssignment.TREATMENT
) -> np.ndarray:
  """Plot the comparison between treatment and control of a candidate design.

  Args:
    pretest_data: pd.DataFrame (date, geo, ...).
    geo_level_eval_data: a pd.DataFrame with columns (geo, response, spend,
      pair)
    response: str, column name used as response in the design.
    spend_proxy: str, column used as spend proxy in the design.
    time_window_for_eval: TimeWindow, representing the time period of pretest
      data used for evaluation of RMSE in estimating iROAS.
    group_control: value representing the control group in the data.
    group_treatment: value representing the treatment group in the data.

  Returns:
    an array of subplots containing the scatterplot and time series comparison
      for the response and spend of the two groups.
  """
  geo_treatment = geo_level_eval_data[geo_level_eval_data['assignment'] ==
                                      group_treatment]
  geo_control = geo_level_eval_data[geo_level_eval_data['assignment'] ==
                                    group_control]
  treatment_geo = geo_treatment['geo'].to_list()
  control_geo = geo_control['geo'].to_list()

  treatment_time_series = pretest_data[pretest_data['geo'].isin(
      treatment_geo)].groupby(
          'date', as_index=False)[[response, spend_proxy]].sum()

  control_time_series = pretest_data[pretest_data['geo'].isin(
      control_geo)].groupby(
          'date', as_index=False)[[response, spend_proxy]].sum()

  _, axes = plt.subplots(2, 2, figsize=(15, 7.5))

  sns.regplot(
      x=np.sqrt(geo_treatment['response']),
      y=np.sqrt(geo_control['response']),
      ax=axes[0, 0],
      fit_reg=False)
  axes[0, 0].set_title(response + ' (in square root)')
  axes[0, 0].set_xlabel('treatment')
  axes[0, 0].set_ylabel('control')
  lim = np.sqrt([
      min([min(geo_control['response']),
           min(geo_treatment['response'])]) * 0.97,
      max([max(geo_control['response']),
           max(geo_treatment['response'])]) * 1.03
  ])
  axes[0, 0].plot(lim, lim, linestyle='--', color='gray')
  axes[0, 0].set_xlim(lim)
  axes[0, 0].set_ylim(lim)

  sns.regplot(
      x=np.sqrt(geo_treatment['spend']),
      y=np.sqrt(geo_control['spend']),
      ax=axes[0, 1],
      fit_reg=False)
  axes[0, 1].set_title(spend_proxy + ' (in square root)')
  axes[0, 1].set_xlabel('treatment')
  axes[0, 1].set_ylabel('control')
  lim = np.sqrt([
      min([min(geo_control['spend']),
           min(geo_treatment['spend'])]) * 0.97,
      max([max(geo_control['spend']),
           max(geo_treatment['spend'])]) * 1.03
  ])
  axes[0, 1].plot(lim, lim, linestyle='--', color='gray')
  axes[0, 1].set_xlim(lim)
  axes[0, 1].set_ylim(lim)

  treatment_time_series.plot(
      x='date',
      y=response,
      color='black',
      label='treatment',
      ax=axes[1, 0])
  control_time_series.plot(
      x='date', y=response, color='red', label='control', ax=axes[1, 0])

  axes[1, 0].axvline(
      x=time_window_for_eval.first_day,
      color='blue',
      ls='-',
      label='evaluation window')
  axes[1, 0].axvline(x=time_window_for_eval.last_day, color='blue', ls='-')
  axes[1, 0].legend()
  axes[1, 0].set_ylabel(response)
  axes[1, 0].set_xlabel('date')

  treatment_time_series.plot(
      x='date',
      y=spend_proxy,
      color='black',
      label='treatment',
      ax=axes[1, 1])
  control_time_series.plot(
      x='date', y=spend_proxy, color='red', label='control', ax=axes[1, 1])
  axes[1, 1].axvline(
      x=time_window_for_eval.first_day,
      color='blue',
      ls='-',
      label='evaluation window')
  axes[1, 1].axvline(x=time_window_for_eval.last_day, color='blue', ls='-')
  axes[1, 1].legend()
  axes[1, 1].set_ylabel(spend_proxy)
  axes[1, 1].set_xlabel('date')

  return axes


def plot_paired_comparison(
    pretest_data: pd.DataFrame, geo_level_eval_data: pd.DataFrame,
    response: str,
    time_window_for_design: TimeWindow,
    time_window_for_eval: TimeWindow,
    group_control: int = common_classes.GeoAssignment.CONTROL,
    group_treatment: int = common_classes.GeoAssignment.TREATMENT,
    legend_location: str = 'best'
) -> sns.FacetGrid:
  """Plot the time series of the response variable for each pair.

  Args:
    pretest_data: pd.DataFrame (date, geo, ...).
    geo_level_eval_data: a pd.DataFrame with columns (geo, response, spend,
      pair)
    response: str, column name used as response in the design.
    time_window_for_design: TimeWindow, representing the time period of
      pretest data used for the design (training + eval).
    time_window_for_eval: TimeWindow, representing the time period of pretest
      data used for evaluation of RMSE in estimating iROAS.
    group_control: value representing the control group in the data.
    group_treatment: value representing the treatment group in the data.
    legend_location: location to place the legend in each plot. Acceptable
      values are of the form 'upper left', 'lower right', etc.; see the
      documentation at
      https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.legend.html#matplotlib.pyplot.legend.

  Returns:
    g: sns.FacetGrid containing one axis for each pair of geos. Each axis
      contains the time series plot of the response variable for the
      treated geo vs the control geo for a particular pair in the design.
  """
  experiment_geo_list = geo_level_eval_data['geo'].to_list()

  geos_assigned = geo_level_eval_data[['geo', 'pair', 'assignment']]

  temporary = pretest_data[pretest_data['geo'].isin(
      experiment_geo_list)]
  temporary = temporary[temporary['date'].between(
      time_window_for_design.first_day, time_window_for_design.last_day)]
  data_to_plot = pd.merge(
      temporary,
      geo_level_eval_data[['geo', 'pair', 'assignment']],
      on='geo',
      how='left')

  g = sns.FacetGrid(
      data_to_plot,
      col='pair',
      hue='assignment',
      col_wrap=2,
      sharey=False,
      sharex=False,
      legend_out=False,
      height=3,
      aspect=2)
  g = (g.map(plt.plot, 'date', response).add_legend())
  pair_list = sorted(geos_assigned['pair'].unique())
  for ind in range(len(g.axes)):
    pair = geos_assigned['pair'] == pair_list[ind]
    cont = geos_assigned[
        pair
        & (geos_assigned['assignment'] == group_control)]['geo'].values[0]
    treat = geos_assigned[
        pair
        & (geos_assigned['assignment'] == group_treatment)]['geo'].values[0]
    g.axes[ind].axvline(x=time_window_for_eval.last_day, color='black', ls='-')
    g.axes[ind].axvline(x=time_window_for_design.last_day, color='red', ls='--')
    g.axes[ind].axvline(x=time_window_for_eval.first_day, color='black', ls='-')
    g.axes[ind].axvline(
        x=time_window_for_design.first_day, color='red', ls='--')

    g.axes[ind].legend([
        'treatment' + ' (geo {})'.format(treat), 'control' +
        ' (geo {})'.format(cont), 'Evaluation period', 'Training period'
    ],
                       loc=legend_location)

  return g
