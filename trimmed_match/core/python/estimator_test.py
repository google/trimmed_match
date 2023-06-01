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
"""Tests for estimator."""

from trimmed_match.core.python import estimator
import unittest
from pybind11_abseil.pybind11_abseil import status


class EstimatorTest(unittest.TestCase):

  def testTrimmedMatchCppError(self):
    # catches errors from C++ code
    with self.assertRaises(status.StatusNotOk):
      tm = estimator.TrimmedMatch([1, 2, 3, 4],
                                  [-10, -1, 1, 10], 0.25)
      tm.Report(1.281551566, 0.1)

if __name__ == "__main__":
  unittest.main()
