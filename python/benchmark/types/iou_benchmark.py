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

import datetime
import random
import sys
from datetime import datetime
from typing import Sequence

import numpy as np

from rikai.types import Box2d


def iou_matrix_naive_version(box_list1: Sequence[Box2d], box_list2: Sequence[Box2d]) -> np.ndarray:
    def getvalue(boxa: Box2d, boxb: Box2d):
        xmin = max(boxa.xmin, boxb.xmin)
        ymin = max(boxa.ymin, boxb.ymin)
        xmax = min(boxa.xmax, boxb.xmax)
        ymax = min(boxa.ymax, boxb.ymax)
        inter_area = max(0.0, xmax - xmin) * max(0.0, ymax - ymin)
        return inter_area / (boxa.area + boxb.area - inter_area)

    result = list()
    for a in box_list1:
        row = list()
        for b in box_list2:
            row.append(getvalue(a, b))
        result.append(np.asarray(row))
    return np.asarray(result)


def benchmark(list1_len: int, list2_len: int, times: int):
    def a_random_box2d():
        x_min = random.uniform(0, 1)
        y_min = random.uniform(0, 1)
        x_max = random.uniform(x_min, 1)
        y_max = random.uniform(y_min, 1)
        return Box2d(x_min, y_min, x_max, y_max)

    list1 = [a_random_box2d() for _ in range(0, list1_len)]
    list2 = [a_random_box2d() for _ in range(0, list2_len)]

    time0 = datetime.now()
    count = 0
    while count < times:
        count += 1
        iou_matrix_naive_version(list1, list2)

    time1 = datetime.now()
    count = 0
    while count < times:
        count += 1
        Box2d.iou_matrix(list1, list2)

    time2 = datetime.now()
    naive_seconds = (time1 - time0).total_seconds()
    vectorized_seconds = (time2 - time1).total_seconds()
    print("naive method cost {} seconds".format(naive_seconds))
    print("vectorized method cost {} seconds".format(vectorized_seconds))
    print("naive/vectorized {}".format(naive_seconds / vectorized_seconds))


if __name__ == '__main__':
    benchmark(*[int(x) for x in sys.argv[1:]])
