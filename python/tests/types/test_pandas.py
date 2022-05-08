#  Copyright 2022 Rikai Authors
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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

from rikai.types import Image, Box2d
from rikai.types.pandas import ImageDtype, Box2dDtype


def test_image(tmp_path):
    # Convert string to Image
    s = pd.Series(np.array(['s3://bucket/path/to/image.jpg']),
                  dtype=ImageDtype())
    assert s.dtype.name == 'image'
    assert isinstance(s[0], Image)

    df = pd.DataFrame({'c': s})
    assert len(df) == len(s)
    assert df.c.dtype == 'image'

    # Check roundtrip pandas<>arrow
    table = pa.Table.from_pandas(df)
    assert len(table) == len(s)
    arrow_type = table.field('c').type
    assert arrow_type.extension_name == 'rikai.image'
    assert arrow_type.storage_type == pa.struct([
        pa.field('uri', pa.string()),
        pa.field('data', pa.binary())
    ])
    assert arrow_type.to_pandas_dtype().name == 'image'

    df_rt = table.to_pandas()
    assert len(df_rt) == len(s)
    assert df_rt.c.dtype == 'image'

    # Check roundtrip pandas<>parquet
    pq.write_to_dataset(table, str(tmp_path))
    df_from_parquet = pd.read_parquet(str(tmp_path))
    assert len(df_from_parquet) == len(s)
    assert df_from_parquet.c.dtype == 'image'
    assert isinstance(df_from_parquet.c[0], Image)


def test_nested(tmp_path):
    s1 = pd.Series(['s3://bucket/path/to/image.jpg'], dtype='image')
    assert s1.dtype.name == 'image'

    s2 = pd.Series([[{
        'label': 'foo',
        'box': Box2d(0, 0, 100, 100),
        'score': 0.8
    }, {
        'label': 'bar',
        'box': Box2d(0, 0, 100, 200),
        'score': 0.7
    }]])

    df = pd.DataFrame({
        'image': s1,
        'annotations': s2
    })

    table = df.rikai.to_table()
    assert table.field('image').type.extension_name == 'rikai.image'
    box = table.field('annotations').type.value_type['box']
    assert box.type.extension_name == 'rikai.box2d'

    df.rikai.save(str(tmp_path))

    df_rt = pd.DataFrame.rikai.load(str(tmp_path))

    assert df_rt.image.dtype == 'image'
    assert isinstance(df_rt.annotations[0][0]['box'], Box2d)


@pytest.mark.skip("TODO add NA handling")
def test_handle_na():
    arr = pd.Series(['s3://bucket/path/to/image.jpg', None], dtype='image')
    pa.array(arr)  # raises ArrowTypeError
