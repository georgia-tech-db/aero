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
import time

from typing import List

import pandas as pd

from eva.configuration.configuration_manager import ConfigurationManager
from eva.udfs.abstract.abstract_udf import AbstractClassifierUDF


class SyntheticUDF1(AbstractClassifierUDF):
    def setup(self):
        config = ConfigurationManager()
        self.cost = config.get_value("experimental", "synthetic_udf2_cost")
        self.sel = config.get_value("experimental", "synthetic_udf2_sel")

        # Meta-data
        self.call_time = 0
        self.label_list = ["False" for _ in range(10)]
        for i in range(int(10 * self.sel)):
            self.label_list[i] = "True"

    @property
    def name(self) -> str:
        return "SyntheticUDF1"

    @property
    def labels(self) -> List[str]:
        return ["True", "False"]

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        df_size = df.size
        time.sleep(self.cost * df_size)
        outcome = []
        for _ in range(df_size):
            outcome.append({
                "labels": self.label_list[self.call_time]
            })
            self.call_time = (self.call_time + 1) % 10
        return pd.DataFrame(outcome, columns=["labels"])
