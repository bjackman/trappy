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


import unittest
from trappy import utils
import pandas
from pandas.util.testing import assert_series_equal


class TestUtils(unittest.TestCase):

    def test_handle_duplicate_index(self):
        """Test Util Function: handle_duplicate_index
        """

        # Refer to the example in the function doc string
        values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 1.0, 6.0, 7.0]
        series = pandas.Series(values, index=index)
        new_index = [0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0]

        with self.assertRaises(ValueError):
            series.reindex(new_index)

        max_delta = 0.001
        expected_index = [0.0, 1.0, 1 + max_delta, 6.0, 7.0]
        expected_series = pandas.Series(values, index=expected_index)
        series = utils.handle_duplicate_index(series, max_delta)
        assert_series_equal(series, expected_series)

        # Make sure that the reindex doesn't raise ValueError any more
        series.reindex(new_index)

    def test_handle_duplicate_index_duplicate_end(self):
        """handle_duplicate_index copes with duplicates at the end of the series"""

        max_delta = 0.001
        values = [0, 1, 2, 3, 4]
        index = [0.0, 1.0, 2.0, 6.0, 6.0]
        expected_index = index[:]
        expected_index[-1] += max_delta
        series = pandas.Series(values, index=index)
        expected_series = pandas.Series(values, index=expected_index)

        series = utils.handle_duplicate_index(series, max_delta)
        assert_series_equal(series, expected_series)

    def test_squash_into_window_one(self):
        """test_squash_into_window1: one event in window"""
        df = pandas.DataFrame([5, 6, 7, 8, 9], index=range(5))
        df_squashed = utils.squash_into_window(df, (2.1, 2.9))

        expected = pandas.DataFrame([7, 7], index=[2.1, 2.9])

        self.assertTrue((df_squashed[0] == expected[0]).all())

    def test_squash_into_window_multiple(self):
        """test_squash_into_window1: multiple events in window"""
        df = pandas.DataFrame([5, 6, 7, 8, 9], index=range(5))
        df_squashed = utils.squash_into_window(df, (0.5, 2.9))

        expected = pandas.DataFrame([5, 6, 7, 7], index=[0.5, 1, 2, 2.9])

        self.assertTrue((df_squashed[0] == expected[0]).all())

    def test_squash_into_window_none_after(self):
        """test_squash_into_window1: no event after window"""
        df = pandas.DataFrame([5, 6], index=range(2))
        df_squashed = utils.squash_into_window(df, (0.5, 2.9))

        expected = pandas.DataFrame([5, 6, 6], index=[0.5, 1, 2.9])

        self.assertTrue((df_squashed[0] == expected[0]).all())

    def test_squash_into_window_none_before(self):
        """test_squash_into_window1: no event before window"""
        df = pandas.DataFrame([6, 7], index=[1, 2])
        df_squashed = utils.squash_into_window(df, (0.5, 2.9))

        expected = pandas.DataFrame([6, 7, 7], index=[1, 2, 2.9])

        self.assertTrue((df_squashed[0] == expected[0]).all())

    def test_squash_into_window_none_inside(self):
        """test_squash_into_window1: no event inside window"""
        df = pandas.DataFrame([5, 6], index=range(2))
        df_squashed = utils.squash_into_window(df, (0.1, 0.9))

        expected = pandas.DataFrame([5, 5], index=[0.1, 0.9])

        self.assertTrue((df_squashed[0] == expected[0]).all())
