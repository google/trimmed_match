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

import dataclasses
import enum
from typing import Any, Callable, Dict, NamedTuple

import pandas as pd


class GeoAssignment(enum.IntEnum):
  """Defines the values for Treatment/Control assignment."""
  CONTROL = 0
  TREATMENT = 1
  EXCLUDED = -1


class GeoXType(enum.Enum):
  """Defines the types of experimental arms."""
  CONTROL = enum.auto()  # business as usual
  GO_DARK = enum.auto()  # stop ad spend (0 during test, >0 during pretest)
  HEAVY_UP = enum.auto()  # increase ad spend
  HEAVY_DOWN = enum.auto()  # decrease ad spend
  HOLD_BACK = enum.auto()  # start ad spend during test (0 during pretest)
  GO_DARK_TREATMENT_NOT_BAU_CONTROL = enum.auto(
  )  # 0 ad spend in treatment, changed (up/down) ad spend in control


class GeoLevelData(NamedTuple):
  """Geo level data."""
  geo: int
  response: float
  spend: float


class GeoLevelPotentialOutcomes(NamedTuple):
  """Two potential outcomes."""
  controlled: GeoLevelData
  treated: GeoLevelData


@dataclasses.dataclass
class TimeWindow:
  """Defines a time window using first day and last day."""
  first_day: pd.Timestamp
  last_day: pd.Timestamp

  def __post_init__(self):
    if not isinstance(self.first_day, pd.Timestamp):
      self.first_day = pd.Timestamp(self.first_day)
    if not isinstance(self.last_day, pd.Timestamp):
      self.last_day = pd.Timestamp(self.last_day)

    if self.first_day > self.last_day:
      raise ValueError('TimeWindow(): first_day > last_day: {!r}, {!r}'.format(
          self.first_day, self.last_day))

  def contains(self, dates: pd.Series) -> pd.Series:
    """Checks whether dates fall into this time window.

    Args:
        dates: a series of datetime.

    Returns:
        a series of bools.
    """
    return pd.to_datetime(dates).between(self.first_day, self.last_day)

  def is_included_in(self, other) -> bool:
    """Checks whether it falls into the 'other' time window.

    Args:
      other: another TimeWindow.

    Returns:
      True iff it falls into the 'other' time window.
    """
    return (self.first_day >= other.first_day and
            self.last_day <= other.last_day)

  def finds_latest_date_moving_window_in(self,
                                         input_dates: pd.Series) -> pd.Series:
    """Finds the latest dates in input_dates.

    The number of latest dates is determined by the number of distinct dates
    that fall into this time window. For example, if input_dates consists of
    2020-02-01, 2020-02-02, 2020-02-03, 2020-02-04, and the current time window
    is TimeWindow('2020-01-01', '2020-02-02') covering 2 dates in input_dates.
    The results will return the latest 2 dates: 2020-02-03, 2020-02-04.

    Args:
        input_dates: a series of datetime.

    Returns:
        a series of datetime (a subset of distinct entries in input_dates).

    Raises:
        ValueError: if input_dates does not overlap with current time window.
    """
    sorted_dates = input_dates.drop_duplicates().sort_values(ascending=False)
    duration = self.contains(sorted_dates).sum()
    if duration == 0:
      raise ValueError('Overlap between input_dates and current time window.')
    return sorted_dates[:duration]


def time_window_shifted_from(time_window: TimeWindow,
                             new_first_day: pd.Timestamp) -> TimeWindow:
  """Shift a TimeWindow to a new first_day."""
  delta = time_window.last_day - time_window.first_day
  return TimeWindow(new_first_day, new_first_day + delta)


@dataclasses.dataclass
class FormatOptions:
  """Defines the formatting parameters to change the format of a table.

  The class contains three attributes:
    column:column on which to apply the format.
    function: function that changes the format of the cell of a table.
    args: arguments to be passed to the function above or pandas Styling.
  """
  column: str
  function: Callable[..., Any]
  args: Dict[str, Any]

  def __post_init__(self):
    if 'axis' in self.args:
      self.args['column'] = self.column
    else:
      self.args['subset'] = self.column
