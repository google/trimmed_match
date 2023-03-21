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
"""Wrappers for estimators using the same data input.

Input dataframe: (date, geo, response, spend, pair, assignment, period).
"""
import abc
from typing import NamedTuple, Optional, Set, Dict
import pandas as pd

from trimmed_match.design import common_classes
from trimmed_match.post_analysis import trimmed_match_post_analysis

# Default confidence for GeoX 2-sided confidence interval: 0.8
DEFAULT_CONFIDENCE_LEVEL = 0.8
CONTROL = common_classes.GeoAssignment.CONTROL
TREATMENT = common_classes.GeoAssignment.TREATMENT


class ConfidenceInterval(NamedTuple):
  lower: float
  upper: float
  confidence_level: float


class AnalysisResult(NamedTuple):
  """format of the analysis result."""

  point_estimate: float
  confidence_interval: ConfidenceInterval
  std_error: Optional[float] = None
  trim_rate: Optional[float] = None
  trim_rate_cost: Optional[float] = None
  trimmed_pairs_indices: Optional[Set[int]] = None

  def __str__(self):
    return """estimate=%.2f, ci_lower=%.2f, ci_upper=%.2f""" % (
        self.point_estimate, self.confidence_interval.lower,
        self.confidence_interval.upper)

  def to_dict(self) -> Dict[str, float]:
    return {
        'point_estimate': self.point_estimate,
        'conf_interval_lower': self.confidence_interval.lower,
        'conf_interval_upper': self.confidence_interval.upper,
        'conf_level': self.confidence_interval.confidence_level,
        'trim_rate': self.trim_rate,
        'trim_rate_cost': self.trim_rate_cost,
    }


class GeoXEstimator(abc.ABC):
  """Abstraction."""
  budget: float
  confidence_level: float

  def __init__(self, confidence_level: float = DEFAULT_CONFIDENCE_LEVEL):
    self.confidence_level = confidence_level

  @abc.abstractmethod
  def analyze(self,
              geoxts: pd.DataFrame,
              group_control: Optional[int] = CONTROL,
              group_treatment: Optional[int] = TREATMENT) -> AnalysisResult:
    pass

  def set_budget(self, budget: float):
    self.budget = budget

  def __eq__(self, other: 'GeoXEstimator'):
    if not isinstance(other, type(self)):
      raise NotImplementedError(f'Cannot compare instance of {type(self)} with '
                                f'instance of {type(other)}.')
    # Check that all class instance attributes are equal.
    return all(
        getattr(other, attribute) == value
        for attribute, value in self.__dict__.items())


class TrimmedMatch(GeoXEstimator):
  """TrimmedMatch Estimator."""
  trim_rate: float
  max_trim_rate: float

  def __init__(self,
               trim_rate: float = -1.0,
               max_trim_rate: float = 0.25,
               confidence_level: float = 0.8):
    """Constructor.

    Args:
      trim_rate: share of pairs trimmed in the analysis. Has to be between 0 and
        max_trim_rate. The default value of -1.0 indicates that the trim_rate is
        chosen during estimation to minimize the confidence interval half-width
      max_trim_rate: half the largest fraction of pairs that can be trimmed
      confidence_level: confidence level for confidence interval computation
    """
    super().__init__()
    self.trim_rate = trim_rate
    self.max_trim_rate = max_trim_rate

  def analyze(self,
              geoxts: pd.DataFrame,
              group_control: Optional[int] = CONTROL,
              group_treatment: Optional[int] = TREATMENT,
              use_cooldown: bool = False) -> AnalysisResult:
    """Analyze experiment using trimmed match.

    Args:
      geoxts: GeoX Time Series to be analyzed. Requires date, geo, response,
        cost, assignment, period and pair.
      group_control: value representing the control group in the data.
      group_treatment: value representing the treatment group in the data.
      use_cooldown: False (only using test period), True (using test +
        cooldown).

    Returns:
      AnalysisResult with point_estimate and confidence_interval
    """

    if 'pair' not in geoxts.columns:
      raise ValueError('Pairing is required to analyze with trimmed match.')

    analysis_data = trimmed_match_post_analysis.prepare_data_for_post_analysis(
        geox_data=geoxts,
        exclude_cooldown=not use_cooldown,
        group_control=group_control,
        group_treatment=group_treatment)
    results = trimmed_match_post_analysis.calculate_experiment_results(
        data=analysis_data,
        trim_rate=self.trim_rate,
        max_trim_rate=self.max_trim_rate,
        confidence=self.confidence_level)

    trimmed_cost = geoxts.cost[geoxts.pair.isin(results.trimmed_pairs)].sum()
    return AnalysisResult(
        point_estimate=results.report.estimate,
        confidence_interval=ConfidenceInterval(
            lower=results.report.conf_interval_low,
            upper=results.report.conf_interval_up,
            confidence_level=self.confidence_level),
        std_error=results.report.std_error,
        trim_rate=results.report.trim_rate,
        trim_rate_cost=trimmed_cost/geoxts.cost.sum(),
        trimmed_pairs_indices=results.report.trimmed_pairs_indices)
