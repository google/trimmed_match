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

"""A method to generate random geo assignment that passes the balance tests."""

from typing import List
import numpy as np
import pandas as pd

from scipy import stats

from trimmed_match import estimator

_MAX_RANDOM_ASSIGNMENTS = 200


def _binomial_sign_test(delta_responses: List[float],
                        confidence: float = 0.8) -> bool:
  """Returns whether the number of positive pairs is about half of total.

  Args:
    delta_responses: response differences from each geo pair.
    confidence: the confidence level for a two-sided conf. interval.

  Returns:
    True if the number of positive values in delta_response is within the conf.
  interval of Binomial(num_pairs, 0.5), where num_pairs = len(delta_response).
  """
  num_pairs = len(delta_responses)
  conf_interval_upper = stats.binom.ppf(0.5 + 0.5 * confidence, num_pairs, 0.5)
  num_positives = sum(np.array(delta_responses) > 0)
  return (num_positives <= conf_interval_upper and
          num_positives >= num_pairs - conf_interval_upper)


def _trimmed_match_aa_test(delta_responses: List[float],
                           delta_spends: List[float],
                           confidence: float = 0.8) -> bool:
  """Returns whether the number of positive pairs is about half of total.

  Args:
    delta_responses: response differences from each geo pair.
    delta_spends: spend differences from each geo pair.
    confidence: the confidence level for a two-sided conf. interval.

  Returns:
    True if the two-sided conf interval of the trimmed match estimator covers
  0.
  """
  report = estimator.TrimmedMatch(delta_responses,
                                  delta_spends).Report(confidence=confidence)
  return report.conf_interval_up > 0 and report.conf_interval_low < 0


def _generate_random_paired_assignment(num_pairs: int) -> np.array:
  """Generates random paired assignment.

  For each geo pair, the two geos are assigned to (True, False) or (False, True)
  with probability 1/2.

  Args:
    num_pairs: num of geo pairs.

  Returns:
    1-dim np.array of length 2 * num_pairs, with dtype=bool, where first two
    entries make a pair, next two entries make a pair, and so on.

  Raises:
    ValueError: num_pairs is zero or smaller.
  """
  if num_pairs < 1:
    raise ValueError(f'num_pairs must be >=1, but got {num_pairs}')

  geo_assignment = np.zeros(2 * num_pairs, dtype=bool)
  pair_assignment = (np.random.uniform(-1, 1, num_pairs) > 0)
  treatment_geo_indices = [2 * i for i in range(num_pairs)] + pair_assignment
  geo_assignment[treatment_geo_indices] = True

  return geo_assignment


def _calculate_paired_difference(geo_values: np.array,
                                 assignment: np.array) -> np.array:
  """Calculates treatment/control difference in each pair.

  Args:
    geo_values: np.array with ndim=1 and dtype=float.
    assignment: np.array with ndim=1 and dtype=bool, same length as geo_values.

  Returns:
    np.array with dtype=float.

  Raises:
    ValueError: geo_values and assignment differ in dimension or shape.
    ValueError: when assignment is not a 1-dim np.array with dtype=bool and with
      equal number of positives and negatives.
  """
  if assignment.dtype != bool:
    raise ValueError(
        f'assignment.dtype must be bool but got {assignment.dtype}')

  if assignment.ndim != 1:
    raise ValueError(f'assignment.ndim must be 1 but got {assignment.ndim}')

  if len(assignment) % 2 == 1:
    raise ValueError(f'len(assigment) must be even but got {len(assignment)})')

  if sum(assignment) != int(len(assignment) / 2):
    raise ValueError('Number of positives and negatives must be equal, ' +
                     f'but got {sum(assignment)} vs {sum(~assignment)}')

  if geo_values.shape != assignment.shape:
    raise ValueError('geo_values.shape must be the same as assignment, ' +
                     f'but got {geo_values.shape}')

  return geo_values[assignment] - geo_values[~assignment]


def generate_balanced_random_assignment(
    sign_test_data: pd.DataFrame,
    aa_test_data: pd.DataFrame,
    sign_test_confidence: float = 0.20,
    aa_test_confidence: float = 0.80) -> pd.DataFrame:
  """Geo assignment for matched pairs that passes balance tests.

  Two balance tests: sign test where the number of positive pairs is about half
  of total number of pairs; aa test where the confidence interval of the trimmed
  match estimator covers 0.

  Args:
    sign_test_data: same as aa_test_data, but for Binomial sign test.
    aa_test_data: pd.DataFrame with columns (geo, pair, response, spend), for
      testing whetherTrimmed Match CI covers 0.
    sign_test_confidence: float in (0, 1), confidence level for a 2-sided sign
      test; the smaller this value, the closer the number of positive pairs and
      the number of negative pairs.
    aa_test_confidence: same as sign_test_confidence, but for the aa test.

  Returns:
    pd.DataFrame with columns (geo, pair, assignment), where assignment is bool.

  Raises:
    ValueError: any of ['geo', 'pair', 'response', 'spend'] is missing from the
      columns of sign_test_data or aa_test_data.
    ValueError: sign_test_data and aa_test_data differ on (geo, pair).
  """
  testdata_columns = set(['geo', 'pair', 'response', 'spend'])
  if not testdata_columns.issubset(set(sign_test_data.columns)):
    raise ValueError('Columns of sign_test_data must be (geo, pair, response, '
                     'spend), but got {}'.format(','.join(
                         sign_test_data.columns)))
  if not testdata_columns.issubset(set(aa_test_data.columns)):
    raise ValueError('Columns of aa_test_data must be (geo, pair, response, '
                     'spend), but got {}'.format(','.join(
                         aa_test_data.columns)))

  sign_data = sign_test_data.sort_values(
      by=['pair', 'geo'], inplace=False).reset_index()
  aa_data = aa_test_data.sort_values(
      by=['pair', 'geo'], inplace=False).reset_index()
  if not sign_data[['pair', 'geo']].equals(aa_data[['pair', 'geo']]):
    raise ValueError('sign_test_data and aa_test_data differ on (geo, pair)!')

  num_pairs = int(aa_data.shape[0] / 2)
  iter_num = 0
  while True:
    iter_num += 1
    geo_assignment = _generate_random_paired_assignment(num_pairs)

    # binomial sign test
    delta_responses = _calculate_paired_difference(
        np.array(sign_data['response']), geo_assignment)
    if not _binomial_sign_test(delta_responses, sign_test_confidence):
      continue

    # trimmed match aa test
    delta_responses = _calculate_paired_difference(
        np.array(aa_data['response']), geo_assignment)
    delta_spends = _calculate_paired_difference(
        np.array(aa_data['spend']), geo_assignment)
    if _trimmed_match_aa_test(delta_responses, delta_spends,
                              aa_test_confidence):
      break

    if iter_num > _MAX_RANDOM_ASSIGNMENTS:
      raise ValueError('Number of random assignments is above {}'.format(
          _MAX_RANDOM_ASSIGNMENTS))

  output = aa_data[['geo', 'pair']].copy()
  output['assignment'] = list(geo_assignment)

  return output
