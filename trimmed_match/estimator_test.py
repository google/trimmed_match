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

"""Tests for the python CLIF wrapper for trimmedmatch."""

from absl.testing import absltest
from absl.testing import parameterized

from trimmed_match import estimator


def _ConstructReport(**kwargs):
  """Creates a Report."""
  if kwargs.keys() != set([
      "estimate", "std_error", "trim_rate", "confidence", "conf_interval_low",
      "conf_interval_up", "epsilons", "trimmed_pairs_indices",
      "candidate_results"
  ]):
    raise KeyError("The arguments are incorrect")

  temp_results = []
  for x in kwargs["candidate_results"]:
    temp = estimator.TrimAndError()
    temp.trim_rate = x[0]
    temp.iroas = x[1]
    temp.std_error = x[2]
    temp_results.append(temp)

  return estimator.Report(
      kwargs.get("estimate"), kwargs.get("std_error"), kwargs.get("trim_rate"),
      kwargs.get("confidence"), kwargs.get("conf_interval_low"),
      kwargs.get("conf_interval_up"), kwargs.get("epsilons"),
      kwargs.get("trimmed_pairs_indices"), temp_results)


class EstimatorTest(absltest.TestCase):

  def setUp(self):
    """Setup reusable data."""
    super(EstimatorTest, self).setUp()

    # test example
    self._delta_response = [1.0, 10.0, 3.0, 8.0]
    self._delta_cost = [1.0, 5.0, 2.0, 5.0]
    self._iroas0 = sum(self._delta_response) / sum(self._delta_cost)
    self._iroas1 = sum(self._delta_response[2:]) / sum(self._delta_cost[2:])

    self._report_no_trim = _ConstructReport(
        estimate=self._iroas0,
        std_error=0.138,
        trim_rate=0.0,
        confidence=0.9,
        conf_interval_low=1.250,
        conf_interval_up=1.905,
        epsilons=[
            self._delta_response[i] - self._delta_cost[i] * self._iroas0
            for i in range(0, len(self._delta_response))
        ],
        trimmed_pairs_indices=set(),
        candidate_results=[[0, 1.692, 0.138]])

    self._report_trim1 = _ConstructReport(
        estimate=self._iroas1,
        std_error=0.041,
        trim_rate=0.20,
        confidence=0.9,
        conf_interval_low=-29.374,
        conf_interval_up=1.619,
        epsilons=[
            self._delta_response[i] - self._delta_cost[i] * self._iroas1
            for i in range(0, len(self._delta_response))
        ],
        trimmed_pairs_indices=set([0, 1]),
        candidate_results=[[0.20, 1.571, 0.041]])

    self._report_auto_trim = _ConstructReport(
        estimate=self._iroas1,
        std_error=0.041,
        trim_rate=0.25,
        confidence=0.9,
        conf_interval_low=-29.374,
        conf_interval_up=1.619,
        epsilons=[
            self._delta_response[i] - self._delta_cost[i] * self._iroas1
            for i in range(0, len(self._delta_response))
        ],
        trimmed_pairs_indices=set([0, 1]),
        candidate_results=[[0.0, 1.692, 0.138], [0.25, 1.571, 0.041]])

  def AssertReportEqual(self, expected, report):
    """Compares Report with expected values."""
    for attr in [
        "estimate", "std_error", "trim_rate", "confidence", "conf_interval_low",
        "conf_interval_up"
    ]:
      self.assertAlmostEqual(
          getattr(expected, attr), getattr(report, attr), places=3)
    self.assertEqual(
        len(expected.candidate_results), len(report.candidate_results))
    self.assertEqual(len(expected.epsilons), len(report.epsilons))
    for e1, e2 in zip(expected.epsilons, report.epsilons):
      self.assertAlmostEqual(e1, e2, places=3)
    self.assertSetEqual(expected.trimmed_pairs_indices,
                        report.trimmed_pairs_indices)
    for i in range(0, len(expected.candidate_results)):
      for attr in ["trim_rate", "iroas", "std_error"]:
        self.assertAlmostEqual(
            getattr(expected.candidate_results[i], attr),
            getattr(report.candidate_results[i], attr),
            places=3)

  def testTrimmedMatchValueError(self):
    # if max_trim_rate is negative
    with self.assertRaisesRegex(ValueError, "max_trim_rate is negative."):
      _ = estimator.TrimmedMatch(self._delta_response, self._delta_cost, -0.1)
    # if delta_response and delta_delta have different lengths
    with self.assertRaisesRegex(
        ValueError, "Lengths of delta_response and delta_spend differ."):
      _ = estimator.TrimmedMatch(self._delta_response, self._delta_cost + [1.0])
    # if confidence is outside of (0, 1]
    tm = estimator.TrimmedMatch(self._delta_response, self._delta_cost)
    with self.assertRaisesRegex(ValueError,
                                r"Confidence is outside of \(0, 1\]"):
      _ = tm.Report(-0.5, 0.0)

  def testCalculateEpsilons(self):
    """Tests _CalculateEpsilons."""
    tm = estimator.TrimmedMatch(self._delta_response, self._delta_cost, 0.25)
    report = tm.Report(0.90, 0.0)
    expected = [
        self._delta_response[i] - self._iroas0 * self._delta_cost[i]
        for i in range(0, len(self._delta_response))
    ]
    self.assertEqual(len(expected), len(report.epsilons))
    for i in range(0, len(expected)):
      self.assertAlmostEqual(expected[i], report.epsilons[i], places=3)

  def testReporValueError(self):
    tm = estimator.TrimmedMatch(self._delta_response, self._delta_cost, 0.25)

    @parameterized.parameters((-0.1, 0.1), (1.1, 0.1), (0.8, 0.5))
    def _(self, confidence, trim_rate):
      with self.assertRaises(ValueError):
        tm.Report(confidence, trim_rate)

  def testTrimmedMatchCase(self):
    """Tests with various trim rates."""
    tm = estimator.TrimmedMatch(self._delta_response, self._delta_cost, 0.25)

    @parameterized.parameters((self._report_no_trim, 0.0),
                              (self._report_trim1, 0.20),
                              (self._report_auto_trim, -1))
    def _(self, expected, trim_rate):
      self.AssertReportEqual(expected, tm.Report(0.90, trim_rate))

  def testTrimmedMatchTiedSpend(self):
    tm = estimator.TrimmedMatch([1, 2, 3, 4, 5], [1, 1, 1, 1, 1])
    self.assertAlmostEqual(tm.Report().estimate, 3.0)

  def testTrimmedMatchTiedThetas(self):
    tm = estimator.TrimmedMatch([1, 2, 3, 4, 5], [1, 2, 3, 4, 5])
    self.assertAlmostEqual(tm.Report().estimate, 1.0)

  def testTrimmedMatchTiedThetasConstant(self):
    tm = estimator.TrimmedMatch([1, 1, 1, 1, 1], [1, 1, 1, 1, 1])
    self.assertAlmostEqual(tm.Report().estimate, 1.0)

  def testTrimmedMatchZeroSpend(self):
    with self.assertRaisesRegex(ValueError,
                                "delta_spends are all too close to 0!"):
      _ = estimator.TrimmedMatch([1, 2, 3, 4, 5], [0, 0, 0, 0, 0])


if __name__ == "__main__":
  absltest.main()
