#  Copyright 2021 Rikai Authors
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Helper functions for using :py:func:`assert` with pytests
"""

from typing import Iterable


def assert_count_equal(first: Iterable, second: Iterable, msg=None):
    """Assert ``first`` has the same elements as ``second``, regardless of
    the order.

    See Also
    --------
    :py:meth:`unittest.TestCase.assertCountEqual`
    """
    from unittest import TestCase

    TestCase().assertCountEqual(first, second, msg=msg)
