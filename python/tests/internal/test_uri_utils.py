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


import unittest
from urllib.parse import urldefrag

from rikai.internal.uri_utils import uri_equal


class TestURIUtils(unittest.TestCase):
    def test_uri_equal(self):
        self.assertTrue(uri_equal("/abc/def", "file:///abc/def"))
        self.assertTrue(uri_equal("s3://abc/bar", "s3://abc/bar"))
        self.assertFalse(uri_equal("s3://foo/bar", "gs://foo/bar"))
