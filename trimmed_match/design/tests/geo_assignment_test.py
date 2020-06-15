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

import numpy as np
import pandas as pd

from trimmed_match.design import geo_assignment
import unittest

sign_test = geo_assignment._binomial_sign_test
iroas_test = geo_assignment._trimmed_match_aa_test
generate_assignment = geo_assignment.generate_balanced_random_assignment
generate_paired_assignment = geo_assignment._generate_random_paired_assignment
calculate_paired_difference = geo_assignment._calculate_paired_difference


class TestSupportingFunctions(unittest.TestCase):

  def testBinomialSignAaTest(self):
    self.assertTrue(sign_test([-1, -1, 1], 0.6))
    self.assertFalse(sign_test([-1, -1, -1], 0.6))

  def testTrimmedMatchAaTest(self):
    self.assertFalse(iroas_test([1, 2, 3], [1, 2, 3], 0.8))
    self.assertTrue(iroas_test([0, -1, 1], [1, 2, 3], 0.8))

  def testGenerateRandomPairedAssignment(self):
    with self.assertRaises(ValueError):
      _ = generate_paired_assignment(-1)

    assignment = generate_paired_assignment(10)
    for i in range(10):
      self.assertEqual(assignment[2 * i] + assignment[2 * i + 1], 1)

  def testCalculatePairedDifferenceValueError(self):
    # assignment: not bool, not 1-dim, or odd length
    with self.assertRaises(ValueError):
      _ = calculate_paired_difference(np.array([1, 2]), np.array([True, True]))

    with self.assertRaises(ValueError):
      _ = calculate_paired_difference(np.array([1, 2]), np.array([1.0, 0.0]))

    with self.assertRaises(ValueError):
      _ = calculate_paired_difference(
          np.array([1, 2]), np.array([[True], [False]]))

    with self.assertRaises(ValueError):
      _ = calculate_paired_difference(
          np.array([1, 2]), np.array([True, True, False]))

    # geo_values and assignment differ in shapes
    with self.assertRaises(ValueError):
      _ = calculate_paired_difference(
          np.array([1, 2, 3]), np.array([0, 1], dtype=bool))

  def testCalculatePairedDifferenceSuccess(self):
    geo_values = np.array([1, 2, 3, 4])
    assignment = np.array([True, False, True, False])
    result = calculate_paired_difference(geo_values, assignment)
    np.testing.assert_array_equal(result, np.array([-1, -1]))


class GeoAssignmentTest(unittest.TestCase):

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

  def testGenerateBalancedRandomAssignmentValueError(self):

    # Incorrect columns in aa_test_data
    with self.assertRaises(ValueError):
      _ = generate_assignment(
          self._sign_test_data,
          self._aa_test_data.drop(columns=['spend'], axis=1))

    # Incorrect columns in sign_test_data
    with self.assertRaises(ValueError):
      _ = generate_assignment(
          self._sign_test_data.drop(columns=['spend'], axis=1),
          self._aa_test_data)

    # Mismatch on the number of geos
    with self.assertRaises(ValueError):
      _ = generate_assignment(
          self._sign_test_data,
          self._aa_test_data.append(
              pd.DataFrame({
                  'geo': [10],
                  'pair': [10],
                  'response': [10],
                  'spend': 10
              })))

    # Two test data differ on (geo, pair)
    with self.assertRaises(ValueError):
      aa_test_data = self._aa_test_data.copy()
      aa_test_data['geo'] = [3, 1, 2, 4, 5, 6]
      _ = generate_assignment(self._sign_test_data, aa_test_data)

  def testGenerateBalancedRandomAssignmentSuccess(self):
    balanced_assignment = generate_assignment(self._sign_test_data,
                                              self._aa_test_data)
    assignment = np.array(balanced_assignment['assignment'])

    # passes binomial sign test
    self.assertTrue(
        sign_test(
            calculate_paired_difference(
                np.array(self._sign_test_data['response']), assignment), 0.8))

    # passes trimmed match aa test
    delta_responses = calculate_paired_difference(
        np.array(self._aa_test_data['response']), assignment)
    delta_spends = calculate_paired_difference(
        np.array(self._aa_test_data['spend']), assignment)
    self.assertTrue(iroas_test(delta_responses, delta_spends, 0.8))


if __name__ == '__main__':
  unittest.main()
