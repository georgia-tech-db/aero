# coding=utf-8
# Copyright 2018-2022 EVA
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
import cv2
import numpy as np
import pandas as pd

from eva.executor.executor_utils import ExecutorError
from eva.udfs.abstract.abstract_udf import AbstractUDF
from eva.utils.logging_manager import logger


def COLOR_MAP(hsv_color):
    # Color name and its lower and upper limit in the HSV space.
    COLOR_NAME_2_BOUND = {
        # White / black colors.
        "black": [
            np.uint8([0, 0, 0]),
            np.uint8([180, 255, 30]),
        ],
        "white": [
            np.uint8([0, 0, 231]),
            np.uint8([180, 18, 255]),
        ],
        "gray": [
            np.uint8([0, 0, 40]),
            np.uint8([180, 18, 230]),
        ],
        # Other colors.
        "red": [
            np.uint8([0, 50, 70]),
            np.uint8([9, 255, 255]),
        ],
        "orange": [
            np.uint8([10, 50, 70]),
            np.uint8([24, 255, 255]),
        ],
        "yellow": [
            np.uint8([25, 50, 70]),
            np.uint8([35, 255, 255]),
        ],
        "green": [
            np.uint8([36, 50, 70]),
            np.uint8([89, 255, 255]),
        ],
        "blue": [
            np.uint8([90, 50, 70]),
            np.uint8([128, 255, 255]),
        ],
        "purple": [
            np.uint8([129, 50, 70]),
            np.uint8([158, 255, 255]),
        ],
        "pink": [
            np.uint8([159, 50, 70]),
            np.uint8([180, 255, 255]),
        ],
    }

    for color_name, bound in COLOR_NAME_2_BOUND.items():
        lower, upper = bound
        if np.all(np.less_equal(lower, hsv_color)) and np.all(
            np.greater_equal(upper, hsv_color)
        ):
            return color_name
    return "else"


class Color(AbstractUDF):
    def setup(self):
        self._data_cache = dict()
        self.tot = 0
        self.count = 0

    @property
    def name(self):
        return "Color"

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        from time import perf_counter
        st = perf_counter()

        def _color(row: pd.Series) -> str:
            img = row[0]
            hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            resized_hsv_img = cv2.resize(hsv_img, (1, 1))
            hsv = resized_hsv_img[0, 0]
            return COLOR_MAP(hsv)

        ret = pd.DataFrame()
        ret["color"] = df.apply(_color, axis=1)

        self.tot += perf_counter() - st
        self.count += len(df)
        
        print(f"Color Cost: {self.tot:.3f}, Color #data proceesed: {self.count}")

        return ret
