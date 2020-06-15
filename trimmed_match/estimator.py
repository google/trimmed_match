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

"""The python library to implement the Trimmed Match estimator.

See the tech details in https://ai.google/research/pubs/pub48448/.
"""

from typing import List, Set

import dataclasses
import numpy as np
from scipy import stats
from trimmed_match.core.python import estimator_ext

# A class to report the Trimmed Match estimator for a fixed trim rate:
# trim_rate: float
# iroas: float
# std_error: float
TrimAndError = estimator_ext.TrimAndError


@dataclasses.dataclass
class Report:
  """A class to report the Trimmed Match estimator.

  Attributes:
    estimate: float, point estimate.
    std_error: float, standard error.
    trim_rate: float, trim rate.
    confidence: float, confidence level.
    conf_interval_low: float, lower bound of the confidence interval.
    conf_interval_up: float, upper bound of the confidence interval.
    epsilons: List[float], difference of uninfluenced responses.
    trimmed_pairs_indices: Set[int], the indices of trimmed pairs.
    candidate_results: List[TrimAndError], result for each candidate trim rate.
  """

  estimate: float
  std_error: float
  trim_rate: float
  confidence: float
  conf_interval_low: float
  conf_interval_up: float
  epsilons: List[float]
  trimmed_pairs_indices: Set[int]
  candidate_results: List["TrimAndError"]

  def __str__(self) -> str:
    """Returns a humanized textual representation of the object."""

    return """estimate=%.5f,
              std_error=%.5f,
              trim_rate=%.5f,
              confidence=%.2f,
              conf_interval_low=%.5f,
              conf_interval_up=%.5f""" % (self.estimate, self.std_error,
                                          self.trim_rate, self.confidence,
                                          self.conf_interval_low,
                                          self.conf_interval_up)


class TrimmedMatch(object):
  """The TrimmedMatch estimator.

  Example usage:
    delta_response = [1, 10, 3, 8]
    delta_spend = [1, 5, 2, 5]
    max_trim_rate = 0.25
    tm = TrimmedMatch(delta_response, delta_spend, max_trim_rate)
    report = tm.Report()
  """

  def __init__(self,
               delta_response: List[float],
               delta_spend: List[float],
               max_trim_rate: float = 0.25):
    """Initializes the class.

    Args:
      delta_response: List[float], response difference between the treatment geo
        and the control geo for each pair.
      delta_spend: List[float], spend difference for each pair.
      max_trim_rate: float, default 0.25.

    Raises:
      ValueError: the lengths of delta_response and delta_spend differ, or
      max_trim_rate is negative.
    """
    if len(delta_response) != len(delta_spend):
      raise ValueError("Lengths of delta_response and delta_spend differ.")

    if max_trim_rate < 0.0:
      raise ValueError("max_trim_rate is negative.")

    self._delta_response = delta_response
    self._delta_spend = delta_spend
    self._tm = estimator_ext.TrimmedMatch(
        delta_response, delta_spend,
        min(0.5 - 1.0 / len(delta_response), max_trim_rate))

  def _CalculateEpsilons(self, iroas: float) -> List[float]:
    """Calculates delta_response - delta_cost * iroas."""
    epsilons = []
    zip_two_deltas = zip(self._delta_response, self._delta_spend)
    for delta1, delta2 in zip_two_deltas:
      epsilons.append(delta1 - delta2 * iroas)
    return epsilons

  def Report(self, confidence: float = 0.80, trim_rate: float = -1.0) -> Report:
    """Reports the Trimmed Match estimation.

    Args:
      confidence: float, the confidence level for the two-sided confidence
        interval, default 0.8.
      trim_rate: float, trim rate, a value outside [0, max_trim_rate) triggers
        the data-driven choice described in the Trimmed Match paper.

    Returns:
      Report, as defined in the class Report above.

    Raises:
      ValueError: confidence is outside of (0, 1].
    """
    if (confidence <= 0.0) | (confidence > 1.0):
      raise ValueError("Confidence is outside of (0, 1]")

    output = self._tm.Report(stats.norm.ppf(0.5 + 0.5 * confidence), trim_rate)
    epsilons = self._CalculateEpsilons(output.estimate)
    temp = np.array(epsilons).argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(temp))
    num_pairs = len(ranks)
    left_trim = np.ceil(num_pairs * output.trim_rate)
    trimmed_pairs_indices = set([
        i for i in np.arange(len(ranks))
        if (ranks[i] < left_trim) or (ranks[i] > num_pairs - left_trim - 1)
    ])
    return Report(output.estimate, output.std_error, output.trim_rate,
                  confidence, output.conf_interval_low, output.conf_interval_up,
                  epsilons, trimmed_pairs_indices, output.candidate_results)
