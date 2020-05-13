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

"""A few common classes to be used for the library."""

import enum

from typing import NamedTuple

import pandas as pd


class GeoXType(enum.Enum):
  """Defines the types of experimental arms."""
  CONTROL = enum.auto()  # business as usual
  GO_DARK = enum.auto()  # stop ad spend (0 during test, >0 during pretest)
  HEAVY_UP = enum.auto()  # increase ad spend
  HEAVY_DOWN = enum.auto()  # decrease ad spend
  HOLD_BACK = enum.auto()  # start ad spend during test (0 during pretest)


class GeoLevelData(NamedTuple):
  """Geo level data."""
  geo: int
  response: float
  spend: float


class GeoLevelPotentialOutcomes(NamedTuple):
  """Two potential outcomes."""
  controlled: GeoLevelData
  treated: GeoLevelData


class TimeWindow(object):
  """Defines a time window using first day and last day."""

  def __init__(self, first_day: pd.Timestamp, last_day: pd.Timestamp):
    if first_day >= last_day:
      raise ValueError("TimeWindow(): first_day >= last_day: {!r}, {!r}".format(
          first_day, last_day))
    self.first_day = first_day
    self.last_day = last_day
