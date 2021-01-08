#  Copyright 2020 Rikai Authors
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

from unittest import TestCase

import numpy as np

from rikai.convert import PortableDataType


class PortableDataTypeTest(TestCase):
    def test_numpy_type_to_enum(self):
        matches = [
            (PortableDataType.UINT_8, np.uint8),
            (PortableDataType.UINT_16, np.uint16),
            (PortableDataType.UINT_32, np.uint32),
            (PortableDataType.UINT_64, np.uint64),
            (PortableDataType.INT_8, np.int8),
            (PortableDataType.INT_16, np.int16),
            (PortableDataType.INT_32, np.int32),
            (PortableDataType.INT_64, np.int64),
        ]
        for p_dt, np_dt in matches:
            self.assertEqual(p_dt, PortableDataType.from_numpy(np_dt))
