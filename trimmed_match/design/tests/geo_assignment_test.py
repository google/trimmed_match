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
"""Tests for trimmed_match.design.tests.util."""
from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import pandas as pd

from trimmed_match.design import geo_assignment

sign_test = geo_assignment.binomial_sign_test
iroas_test = geo_assignment.trimmed_match_aa_test
representativeness_test = geo_assignment.representativeness_test
generate_assignment = geo_assignment.generate_balanced_random_assignment
generate_paired_assignment = geo_assignment._generate_random_paired_assignment
calculate_paired_difference = geo_assignment.calculate_paired_difference


class TestSupportingFunctions(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='Default',
          min_response_per_group=0.0,
          min_number_of_geos_per_group=1,
          max_fraction_largest_geo=1.0,
          expected=True,
      ),
      dict(
          testcase_name='Insufficient response',
          min_response_per_group=4.1,
          min_number_of_geos_per_group=1,
          max_fraction_largest_geo=1.0,
          expected=False,
      ),
      dict(
          testcase_name='Insufficient geos',
          min_response_per_group=0.0,
          min_number_of_geos_per_group=3,
          max_fraction_largest_geo=1.0,
          expected=False,
      ),
      dict(
          testcase_name='Largest geo too large',
          min_response_per_group=0.0,
          min_number_of_geos_per_group=1,
          max_fraction_largest_geo=0.7,
          expected=False,
      ),
  )
  def testRepresentativenessTest(
      self,
      min_response_per_group,
      min_number_of_geos_per_group,
      max_fraction_largest_geo,
      expected,
  ):
    geo_values = np.array([1, 2, 3, 4])
    assignment = np.array([True, False, True, False])
    self.assertEqual(
        representativeness_test(
            geo_values,
            assignment,
            min_response_per_group,
            min_number_of_geos_per_group,
            max_fraction_largest_geo,
        ),
        expected,
    )

  def testBinomialSignAaTest(self):
    self.assertTrue(sign_test(np.array([-1, -1, 1]), 0.6))
    self.assertFalse(sign_test(np.array([-1, -1, -1]), 0.6))
    self.assertTrue(sign_test(np.array([0, 0, 0]), 0.6))

  def testTrimmedMatchAaTest(self):
    self.assertFalse(iroas_test(np.array([1, 2, 3]), np.array([1, 2, 3]), 0.8))
    self.assertTrue(iroas_test(np.array([0, -1, 1]), np.array([1, 2, 3]), 0.8))

  def testGenerateRandomPairedAssignment(self):
    with self.assertRaisesRegex(
        ValueError,
        'num_pairs must be >=1, but got -1',
    ):
      _ = generate_paired_assignment(-1)

    assignment = generate_paired_assignment(10)
    for i in range(10):
      self.assertEqual(assignment[2 * i] + assignment[2 * i + 1], 1)

  @parameterized.named_parameters(
      (
          'Unequal number of positives and negatives',
          np.array([1, 2]),
          np.array([True, True]),
          'Number of positives and negatives must be equal, but got 2 vs 0',
      ),
      (
          'Wrong assignment dtype',
          np.array([1, 2]),
          np.array([1.0, 0.0]),
          'assignment.dtype must be bool but got float64',
      ),
      (
          'Wrong assignment dimension',
          np.array([1, 2]),
          np.array([[True], [False]]),
          'assignment.ndim must be 1 but got 2',
      ),
      (
          'Length of assignment is odd',
          np.array([1, 2]),
          np.array([True, True, False]),
          r'len\(assignment\) must be even but got 3',
      ),
      (
          'Length of geo_values and assignment differ',
          np.array([1, 2, 3]),
          np.array([0, 1], dtype=bool),
          r'geo_values.shape must be the same as assignment, but got \(3,\)',
      ),
  )
  def testCalculatePairedDifferenceValueError(
      self, geo_values, assignment, error
  ):
    with self.assertRaisesRegex(ValueError, error):
      calculate_paired_difference(geo_values, assignment)

  def testCalculatePairedDifferenceSuccess(self):
    geo_values = np.array([1, 2, 3, 4])
    assignment = np.array([True, False, True, False])
    result = calculate_paired_difference(geo_values, assignment)
    np.testing.assert_array_equal(result, np.array([-1, -1]))


class GeoAssignmentTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._sign_test_data = pd.DataFrame({
        'geo': [1, 2, 3, 4, 5, 6],
        'pair': [1, 1, 2, 2, 3, 3],
        'response': [1, 2, 3, 4, 5, 6],
        'spend': [1, 2, 3, 4, 5, 6]
    })
    self._aa_test_data = pd.DataFrame({
        'geo': [1, 2, 3, 4, 5, 6],
        'pair': [1, 1, 2, 2, 3, 3],
        'response': [1, 2, 3, 4, 5, 6],
        'spend': [1, 1, 2, 4, 5, 6]
    })
    self._representativeness_test_data = pd.DataFrame({
        'geo': [1, 2, 3, 4, 5, 6],
        'pair': [1, 1, 2, 2, 3, 3],
        'response': [1, 2, 3, 4, 5, 6]
    })

  def testGenerateBalancedRandomAssignmentValueError(self):

    # Incorrect columns in aa_test_data
    with self.assertRaisesRegex(
        ValueError,
        r'Columns of aa_test_data must be \(geo, pair, response, spend\), but'
        ' got geo,pair,response',
    ):
      _ = generate_assignment(
          self._representativeness_test_data,
          self._sign_test_data,
          self._aa_test_data.drop(columns=['spend'], axis=1))

    # Incorrect columns in sign_test_data
    with self.assertRaisesRegex(
        ValueError,
        r'Columns of sign_test_data must be \(geo, pair, response, spend\), but'
        ' got geo,pair,response',
    ):
      _ = generate_assignment(
          self._representativeness_test_data,
          self._sign_test_data.drop(columns=['spend'], axis=1),
          self._aa_test_data)

    # Incorrect columns in representativeness_test_data
    with self.assertRaisesRegex(
        ValueError,
        r'Columns of representativeness_test_data must be \(geo, pair,'
        r' response\), but got geo,pair',
    ):
      _ = generate_assignment(
          self._representativeness_test_data.drop(columns=['response'], axis=1),
          self._sign_test_data,
          self._aa_test_data)

    # Mismatch on the number of geos
    with self.assertRaisesRegex(
        ValueError,
        r'sign_test_data and aa_test_data differ on \(geo, pair\)',
    ):
      _ = generate_assignment(
          self._representativeness_test_data,
          self._sign_test_data,
          pd.concat([
              self._aa_test_data,
              pd.DataFrame(
                  {'geo': [10], 'pair': [10], 'response': [10], 'spend': 10}
              ),
          ]),
      )

    # Two test data differ on (geo, pair)
    with self.assertRaisesRegex(
        ValueError,
        r'sign_test_data and aa_test_data differ on \(geo, pair\)',
    ):
      aa_test_data = self._aa_test_data.copy()
      aa_test_data['geo'] = [3, 1, 2, 4, 5, 6]
      _ = generate_assignment(
          self._representativeness_test_data, self._sign_test_data, aa_test_data
      )

    # _representativeness_test_data differ from other test data on (geo, pair)
    with self.assertRaisesRegex(
        ValueError,
        r'sign_test_data and representativeness_data differ on \(geo, pair\)',
    ):
      representativeness_test_data = self._representativeness_test_data.copy()
      representativeness_test_data['geo'] = [3, 1, 2, 4, 5, 6]
      _ = generate_assignment(
          representativeness_test_data, self._sign_test_data, self._aa_test_data
      )

  def testGenerateBalancedRandomAssignmentSuccess(self):
    balanced_assignment = generate_assignment(
        self._representativeness_test_data,
        self._sign_test_data,
        self._aa_test_data,
    )
    assignment = np.array(balanced_assignment['assignment'])

    # passes binomial sign test
    self.assertTrue(
        sign_test(
            calculate_paired_difference(
                np.array(self._sign_test_data['response']), assignment
            ),
            0.2,
        )
    )

    # passes trimmed match aa test
    delta_responses = calculate_paired_difference(
        np.array(self._aa_test_data['response']), assignment
    )
    delta_spends = calculate_paired_difference(
        np.array(self._aa_test_data['spend']), assignment
    )
    self.assertTrue(iroas_test(delta_responses, delta_spends, 0.8))

    # passes representativeness_test
    self.assertTrue(
        representativeness_test(
            self._representativeness_test_data['response'], assignment
        )
    )

  def testGenerateBalancedRandomAssignmentWithTies(self):
    aa_data = pd.DataFrame({
        'geo': [1, 2, 3, 4, 5, 6],
        'pair': [1, 1, 2, 2, 3, 3],
        'response': [1, 1, 1, 1, 1, 2],
        'spend': [1, 1, 1, 1, 1, 2],
    })
    assignment = generate_assignment(
        self._representativeness_test_data, aa_data, aa_data
    )['assignment']
    delta_responses = calculate_paired_difference(
        np.array(aa_data['response']), assignment
    )
    delta_spends = (
        calculate_paired_difference(np.array(aa_data['spend']), assignment)
        + geo_assignment._INCREMENTAL_SPEND_RATIO_FOR_AA_TEST
    )
    self.assertTrue(iroas_test(delta_responses, delta_spends, 0.8))


if __name__ == '__main__':
  absltest.main()
