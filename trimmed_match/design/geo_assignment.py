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

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from trimmed_match import estimator


_MAX_RANDOM_ASSIGNMENTS = 200

# TODO(b/186020999): pass the actual budget
_INCREMENTAL_SPEND_RATIO_FOR_AA_TEST = 0.50


class NoAssignmentError(Exception):
  """An error raised when no valid balanced assignment was found."""


def representativeness_test(
    geo_values: np.ndarray,
    assignment: np.ndarray,
    min_response_per_group: float = 0.0,
    min_number_of_geos_per_group: int = 1,
    max_fraction_largest_geo: float = 1.0
) -> bool:
  """Checks if an assignment passes our conditions for representativeness.

  This function checks whether an assignment passes our proxy conditions for
  representativeness.

  Args:
    geo_values: [number of geos]-np.ndarray, the response values for each geo.
    assignment: [number of geos]-np.ndarray, the assignment for each geo.
    min_response_per_group: the threshold on the minimum response in each group.
    min_number_of_geos_per_group: the threshold on the minimum number of geos in
      both groups.
    max_fraction_largest_geo: the threshold on the share of response covered by
      the largest geo within each group.

  Returns:
    True if the assignment satisfies all the following conditions:
      - the number of geos in both groups is >= min_number_of_geos_per_group
      - the minimum response in each group is >= min_response_per_group
      - the share of response covered by the largest geo is both groups is <=
        max_fraction_largest_geo
  """
  n_true = assignment.sum()
  n_false = (~assignment).sum()
  if min(n_true, n_false) < min_number_of_geos_per_group:
    return False

  response_true = geo_values[assignment].sum()
  response_false = geo_values[~assignment].sum()

  if min(response_true, response_false) < min_response_per_group:
    return False

  if (
      max(
          geo_values[assignment].max() / response_true,
          geo_values[~assignment].max() / response_false,
      )
      > max_fraction_largest_geo
  ):
    return False

  return True


def binomial_sign_test(
    delta_responses: np.ndarray, confidence: float = 0.8
) -> bool:
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
  num_negatives = sum(np.array(delta_responses) < 0)
  return (num_positives <= conf_interval_upper and
          num_negatives <= conf_interval_upper)


def trimmed_match_aa_test(delta_responses: np.ndarray,
                          delta_spends: np.ndarray,
                          confidence: float = 0.8) -> bool:
  """Returns whether the confidence interval for Trimmed Match contains zero.

  Args:
    delta_responses: response differences from each geo pair.
    delta_spends: spend differences from each geo pair.
    confidence: the confidence level for a two-sided conf. interval.

  Returns:
    True if the two-sided conf interval of the trimmed match estimator covers
  0.
  """
  report = estimator.TrimmedMatch(
      delta_responses.tolist(), delta_spends.tolist()
  ).Report(confidence=confidence)
  return report.conf_interval_up > 0 and report.conf_interval_low < 0


def _generate_random_paired_assignment(num_pairs: int) -> np.ndarray:
  """Generates random paired assignment.

  For each geo pair, the two geos are assigned to (True, False) or (False, True)
  with probability 1/2.

  Args:
    num_pairs: num of geo pairs.

  Returns:
    [length 2 * num_pairs]-np.ndarray, with dtype=bool, where first two entries
    make a pair, next two entries make a pair, and so on.

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


def calculate_paired_difference(geo_values: np.ndarray,
                                assignment: np.ndarray) -> np.ndarray:
  """Calculates treatment/control difference in each pair.

  Args:
    geo_values: [number of geos]-np.ndarray with dtype=float.
    assignment: [number of geos]-np.ndarray with dtype=bool.

  Returns:
    np.ndarray with dtype=float.

  Raises:
    ValueError: geo_values and assignment differ in dimension or shape.
    ValueError: when assignment is not a [number of geos]-np.ndarray with
      dtype=bool and with equal number of positives and negatives.
  """
  if assignment.dtype != bool:
    raise ValueError(
        f'assignment.dtype must be bool but got {assignment.dtype}')

  if assignment.ndim != 1:
    raise ValueError(f'assignment.ndim must be 1 but got {assignment.ndim}')

  if len(assignment) % 2 == 1:
    raise ValueError(f'len(assignment) must be even but got {len(assignment)}')

  if sum(assignment) != int(len(assignment) / 2):
    raise ValueError('Number of positives and negatives must be equal, ' +
                     f'but got {sum(assignment)} vs {sum(~assignment)}')

  if geo_values.shape != assignment.shape:
    raise ValueError('geo_values.shape must be the same as assignment, ' +
                     f'but got {geo_values.shape}')

  return geo_values[assignment] - geo_values[~assignment]


def generate_balanced_random_assignment(
    representativeness_test_data: pd.DataFrame,
    sign_test_data: pd.DataFrame,
    aa_test_data: pd.DataFrame,
    sign_test_confidence: float = 0.20,
    aa_test_confidence: float = 0.80,
    min_response_per_group: float = 0.0,
    min_number_of_geos_per_group: int = 1,
    max_fraction_largest_geo: float = 1.0,
) -> pd.DataFrame:
  """Geo assignment for matched pairs that passes balance tests.

  This function returns an assignment that passes the following tests:
  - One representativeness test: checks some conditions as a proxy for
    representativeness.
  - Two balance tests: sign test where the number of positive pairs is about
    half of total number of pairs; aa test where the confidence interval of the
    trimmed match estimator covers 0.

  Args:
    representativeness_test_data: pd.DataFrame with columns (geo, pair,
    response).
    sign_test_data: same as aa_test_data, but for Binomial sign test.
    aa_test_data: pd.DataFrame with columns (geo, pair, response, spend), for
      testing whetherTrimmed Match CI covers 0.
    sign_test_confidence: float in (0, 1), confidence level for a 2-sided sign
      test; the smaller this value, the closer the number of positive pairs and
      the number of negative pairs.
    aa_test_confidence: same as sign_test_confidence, but for the aa test.
    min_response_per_group: float >= 0, the minimum total response in each group
      required in the representativeness test.
    min_number_of_geos_per_group: int > 0, the threshold on the minimum number
      of geos in both groups in the representativeness test.
    max_fraction_largest_geo: float in [0, 1], the threshold on the share of
      response covered by the largest geo within each group in the
      representativeness test.

  Returns:
    pd.DataFrame with columns (geo, pair, assignment), where assignment is bool.

  Raises:
    ValueError: any of ['geo', 'pair', 'response', 'spend'] is missing from the
      columns of sign_test_data or aa_test_data.
    ValueError: any of ['geo', 'pair', 'response'] is missing from the columns
      of representativeness_test_data.
    ValueError: sign_test_data, aa_test_data and representativeness_test_data
      differ on (geo, pair).
    NoAssignmentError: maximum number of assignments reached in trying to find a
      balanced assignment.
  """
  testdata_columns = {'geo', 'pair', 'response', 'spend'}
  if not testdata_columns.issubset(sign_test_data.columns):
    raise ValueError(
        'Columns of sign_test_data must be (geo, pair, response, '
        'spend), but got {}'.format(','.join(sign_test_data.columns))
    )
  if not testdata_columns.issubset(aa_test_data.columns):
    raise ValueError(
        'Columns of aa_test_data must be (geo, pair, response, '
        'spend), but got {}'.format(','.join(aa_test_data.columns))
    )
  if not {'geo', 'pair', 'response'}.issubset(
      representativeness_test_data.columns
  ):
    raise ValueError(
        'Columns of representativeness_test_data must be (geo, pair, response),'
        ' but got {}'.format(','.join(representativeness_test_data.columns))
    )

  representativeness_data = representativeness_test_data.sort_values(
      by=['pair', 'geo'], inplace=False
  ).reset_index(drop=True)
  sign_data = sign_test_data.sort_values(
      by=['pair', 'geo'], inplace=False
  ).reset_index(drop=True)
  aa_data = aa_test_data.sort_values(
      by=['pair', 'geo'], inplace=False
  ).reset_index(drop=True)
  if not sign_data[['pair', 'geo']].equals(aa_data[['pair', 'geo']]):
    raise ValueError('sign_test_data and aa_test_data differ on (geo, pair)!')
  if not sign_data[['pair', 'geo']].equals(
      representativeness_data[['pair', 'geo']]
  ):
    raise ValueError(
        'sign_test_data and representativeness_data differ on (geo, pair)!'
    )

  num_pairs = aa_data.shape[0] // 2
  for iter_num in itertools.count():
    if iter_num >= _MAX_RANDOM_ASSIGNMENTS:
      raise NoAssignmentError(
          'Number of random assignments is above {}'.format(
              _MAX_RANDOM_ASSIGNMENTS
          )
      )

    geo_assignment = _generate_random_paired_assignment(num_pairs)

    # representativeness test
    if not representativeness_test(
        representativeness_data['response'],
        geo_assignment,
        min_response_per_group,
        min_number_of_geos_per_group,
        max_fraction_largest_geo,
    ):
      continue

    # binomial sign test
    delta_responses = calculate_paired_difference(
        np.array(sign_data['response']), geo_assignment
    )
    if not binomial_sign_test(delta_responses, sign_test_confidence):
      continue

    # trimmed match aa test
    delta_responses = calculate_paired_difference(
        np.array(aa_data['response']), geo_assignment
    )
    geo_spends = np.array(aa_data['spend'])
    delta_spends = calculate_paired_difference(geo_spends, geo_assignment) + (
        (geo_spends[geo_assignment] + geo_spends[~geo_assignment])
        * _INCREMENTAL_SPEND_RATIO_FOR_AA_TEST
    )
    if trimmed_match_aa_test(delta_responses, delta_spends, aa_test_confidence):
      break

  output = aa_data[['geo', 'pair']].copy()
  output['assignment'] = list(geo_assignment)

  return output
