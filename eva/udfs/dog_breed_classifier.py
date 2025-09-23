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
from typing import List

import pandas as pd
from PIL import Image
from transformers import pipeline

from eva.udfs.abstract.abstract_udf import AbstractClassifierUDF


class DogBreedClassifier(AbstractClassifierUDF):
    """
    Dog breed classifier.
    """

    def setup(self):
        self.model = pipeline(model="skyau/dog-breed-classifier-vit", device=0)
        self.tot = 0
        self.count = 0

    @property
    def name(self) -> str:
        return "DogBreedClassifier"

    @property
    def labels(self) -> List[str]:
        return []

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        from time import perf_counter
        st = perf_counter()

        outcome = pd.DataFrame()
        df_size = df.size

        for i in range(df_size):
            rgb_img = df.iat[i, 0]
            pil_img = Image.fromarray(rgb_img)
            dog_list = self.model(pil_img)
            top_dog = dog_list[0]
            label = " ".join([lb.lower() for lb in top_dog["label"].split("_")])
            outcome = pd.concat(
                [outcome, pd.DataFrame({"labels": [label]})], ignore_index=True
            )

        self.tot += perf_counter() - st
        self.count += len(df)
        # print(f"Breed Cost: {self.tot:.3f}, Breed #data proceesed: {self.count}")

        return outcome
