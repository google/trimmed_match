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
r"""Generate a simple simulated dataset for the colabs.

Run this script to create two csv files under the folder specified in the flag
"output_dir". The two csv files are called "simulated_design_dataset.csv" and
"simulated_postanalysis_dataset.csv" which can be used for the design and
postanalysis colabs, respectively.
"""

from os import path
from absl import app
from absl import flags

import numpy as np
import pandas as pd


FLAGS = flags.FLAGS
flags.DEFINE_string('start_date', None, 'First date in the time series. '
                    'The date should be in the format YYYY-MM-DD.')
flags.DEFINE_string('end_date', None, 'Last date in the time series. '
                    'The date should be in the format YYYY-MM-DD.')
flags.DEFINE_integer(
    'num_geos', 100, 'number of geos in the simulated dataset')
flags.DEFINE_integer(
    'random_seed', 0, 'base seed for the RNG.')
flags.DEFINE_string('output_dir', '', 'Directory to write results into.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  np.random.seed(FLAGS.random_seed)
  start_date = pd.Timestamp(FLAGS.start_date)
  end_date = pd.Timestamp(FLAGS.end_date)
  max_geos = int((FLAGS.num_geos+2)/2)
  geos = np.random.randint(1, FLAGS.num_geos)
  avg_geo_value = abs(np.random.normal(0, 1, max_geos))
  dates = list(pd.date_range(start=start_date, end=end_date, freq='D'))
  num_weeks = int((end_date - start_date) / np.timedelta64(1, 'W')) + 1
  seasonal_effect = np.sin([2 * np.pi * i / 7 for i in range(len(dates))])
  data = pd.DataFrame(columns=['date', 'geo', 'response', 'cost'])
  for geo in range(max_geos):
    cumulative_error = np.cumsum(np.random.normal(0, 1, len(dates)))
    response = 10000 * np.exp(0.01 * seasonal_effect +
                              0.015 * cumulative_error + avg_geo_value[geo])
    cost = response * 0.01 + abs(np.random.normal(0, 50))
    data = data.append(
        pd.DataFrame({
            'date':
                dates,
            'response':
                response * (1 + 0.05 * np.random.normal(0, 1, len(dates))),
            'geo':
                geo,
            'cost':
                cost
        }),
        ignore_index=True)
    data = data.append(
        pd.DataFrame({
            'date':
                dates,
            'response':
                response * (1 + 0.05 * np.random.normal(0, 1, len(dates))),
            'geo':
                geo + FLAGS.num_geos,
            'cost':
                cost + 0.05 * np.random.normal(0, 1, len(dates))
        }),
        ignore_index=True)

  design_data = pd.DataFrame(data)

  postanalysis_data = design_data.copy()
  pretest = postanalysis_data.groupby('geo', as_index=False).sum()
  geos_ordered = pretest.sort_values(['response', 'geo'],
                                     ascending=[False,
                                                True]).reset_index(drop=True)
  geopairs_left = geos_ordered.iloc[::2, :].reset_index(drop=True)
  geopairs_right = geos_ordered.iloc[1::2, :].reset_index(drop=True)
  dist = (
      abs(geopairs_left['response'] - geopairs_right['response']) /
      sum(geos_ordered['response']))

  geopairs_left = geopairs_left.assign(dist=dist)
  geopairs_right = geopairs_right.assign(dist=dist)
  pairs = (pd.DataFrame({
      'geo1': geopairs_left['geo'],
      'geo2': geopairs_right['geo'],
      'distance': geopairs_left['dist']
  }).sort_values(by=['distance', 'geo1'],
                 ascending=[False, True])).reset_index(drop=True)
  npairs = geopairs_left.shape[0]
  pairs['pair'] = range(1, npairs + 1)

  geo_to_pair = pd.DataFrame({
      'geo': pairs['geo1'].tolist() + pairs['geo2'].tolist(),
      'pair': pairs['pair'].tolist() + pairs['pair'].tolist()
  })

  geo_to_pair = geo_to_pair.sort_values(by='pair').reset_index()
  random_assignment = np.random.uniform(-1, 1, npairs) > 0
  treat = [2 * x for x in range(npairs)] + random_assignment
  geo_to_pair['assignment'] = 2
  geo_to_pair.loc[geo_to_pair.index.isin(treat), 'assignment'] = 1

  postanalysis_data = pd.merge(
      postanalysis_data, geo_to_pair, on='geo', how='left')
  postanalysis_data.drop(
      postanalysis_data[postanalysis_data['pair'] == 1].index, inplace=True)
  design_data.to_csv(path.join(FLAGS.output_dir,
                               'simulated_design_dataset.csv'),
                     index=False)
  postanalysis_data.to_csv(path.join(
     FLAGS.output_dir,
     'simulated_postanalysis_dataset.csv'),
                           index=False)

if __name__ == '__main__':
  app.run(main)
