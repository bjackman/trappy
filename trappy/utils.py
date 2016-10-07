#    Copyright 2015-2016 ARM Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pandas as pd

"""Generic functions that can be used in multiple places in trappy
"""

def listify(to_select):
    """Utitlity function to handle both single and
    list inputs
    """

    if not isinstance(to_select, list):
        to_select = [to_select]

    return to_select

def handle_duplicate_index(data,
                           max_delta=0.000001):
    """Handle duplicate values in index

    :param data: The timeseries input
    :type data: :mod:`pandas.Series`

    :param max_delta: Maximum interval adjustment value that
        will be added to duplicate indices
    :type max_delta: float

    Consider the following case where a series needs to be reindexed
    to a new index (which can be required when different series need to
    be combined and compared):
    ::

        import pandas
        values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 1.0, 6.0, 7.0]
        series = pandas.Series(values, index=index)
        new_index = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0]
        series.reindex(new_index)

    The above code fails with:
    ::

        ValueError: cannot reindex from a duplicate axis

    The function :func:`handle_duplicate_axis` changes the duplicate values
    to
    ::

        >>> import pandas
        >>> from trappy.utils import handle_duplicate_index

        >>> values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 1.0, 6.0, 7.0]
        series = pandas.Series(values, index=index)
        series = handle_duplicate_index(series)
        print series.index.values
        >>> [ 0.        1.        1.000001  6.        7.      ]

    """

    index = data.index
    new_index = index.values

    dups = index.get_duplicates()

    for dup in dups:
        # Leave one of the values intact
        dup_index_left = index.searchsorted(dup, side="left")
        dup_index_right = index.searchsorted(dup, side="right") - 1
        num_dups = dup_index_right - dup_index_left + 1

        # Calculate delta that needs to be added to each duplicate
        # index
        try:
            delta = (index[dup_index_right + 1] - dup) / num_dups
        except IndexError:
            # dup_index_right + 1 is outside of the series (i.e. the
            # dup is at the end of the series).
            delta = max_delta

        # Clamp the maximum delta added to max_delta
        if delta > max_delta:
            delta = max_delta

        # Add a delta to the others
        dup_index_left += 1
        while dup_index_left <= dup_index_right:
            new_index[dup_index_left] += delta
            delta += delta
            dup_index_left += 1

    return data.reindex(new_index)

def squash_into_window(df, window):
    """Take a slice of a series, assuming that values don't change between events

    This function takes a slice of a series, copying the last event that occured
    before the window to the beginning of the window, and the last event inside
    the window to the end of the window. That is, assuming there are, in the
    input, events both before and during the window, the output will have an
    event at the beginning and at the end of the window.

    For example, consider this (sparse) series:

    ====== =======
     Time   Value
    ====== =======
      0      0
      2      1
      8      0
      9      1
     12      1
    ====== =======

    If this series bears the assumption that the value does not change between
    events, it be viewed like this (where '*' depicts events in the series, and
    the dotted line represents the assumed value):

          .. code::

            1            *----------------------------+    *--------------*
                         |                            |    |
            0  *---------+                            *----+
               0    1    2    3    4    5    6    7   8    9    10   11   12

    If this series were naively windowed between time values 5 and 11 it would
    look like this:

          .. code::

            1                                              *----+
                                                           |    |
            0                                         *----+    *
                                        5    6    7   8    9    10   11

                                        | <-------  window  -------> |

    The value is lost between time values 5 - 8 and 10 - 11.

    This function would duplicate the events at times 2 and 10 to occur at times
    5 and 11 respectively, like so:

          .. code::

            1                                              *----+
                                                           |    |
            0                           *-------------*----+    *----*
                                        5    6    7   8    9    10   11

                                        | <-------  window  -------> |

    """
    start_time, end_time = window

    window_df = df[(df.index >= start_time) & (df.index < end_time)]

    # Get a DataFrame containing only the last event occuring before the start
    # of the window
    first_event_df = df[df.index < start_time].iloc[-1:]
    if not first_event_df.empty:
        # Bump the time value for that event up to the beginning of the window.
        first_event_df.index = [start_time]

    df = pd.concat([first_event_df, window_df])

    # Get a DataFrame containing only the last event occuring before the end
    # of the window
    last_event_df = df[df.index <= end_time].iloc[-1:]
    if not last_event_df.empty:
        # Bump the time value for that event up to the end of the window.
        last_event_df.index = [end_time]

    return pd.concat([df, last_event_df])
