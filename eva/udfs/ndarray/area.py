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
import numpy as np
import pandas as pd

from eva.udfs.abstract.abstract_udf import AbstractUDF


class Area(AbstractUDF):
    def setup(self):
        self.tot = 0
        self.count = 0
        pass

    @property
    def name(self):
        return "Area"

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        from time import perf_counter
        st = perf_counter()
        def area(row: pd.Series) -> np.ndarray:
            row = row.to_list()
            frame = row[0]
            bboxes = row[1]

            x0, y0, x1, y1 = np.asarray(bboxes, dtype="int")
            w,h = frame.shape[:2]
            return abs(x1-x0)*abs(y1-y0)/(w*h)

        ret = pd.DataFrame()
        ret["area"] = df.apply(area, axis=1)
        
        self.tot += perf_counter() - st
        self.count += len(df)
        
        print(f"Area Cost: {self.tot:.3f}, Area #data proceesed: {self.count}")
        
        return ret
