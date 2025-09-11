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

import numpy as np
import pandas as pd
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from eva.udfs.abstract.abstract_udf import AbstractClassifierUDF


class SceneClassifier(AbstractClassifierUDF):
    """
    Scene classifier.
    """

    def setup(self):
        pt = hf_hub_download(repo_id="keremberke/yolov8s-scene-classification",
                     filename="best.pt") 
        self.model = pt
        self.device = 0

    @property
    def name(self) -> str:
        return "SceneClassifier"

    @property
    def labels(self) -> List[str]:
        return [
            "airport_inside",
            "artstudio",
            "auditorium",
            "bakery",
            "bookstore",
            "bowling",
            "buffet",
            "casino",
            "children_room",
            "church_inside",
            "classroom",
            "cloister",
            "closet",
            "clothingstore",
            "computerroom",
            "concert_hall",
            "corridor",
            "deli",
            "dentaloffice",
            "dining_room",
            "elevator",
            "fastfood_restaurant",
            "florist",
            "gameroom",
            "garage",
            "greenhouse",
            "grocerystore",
            "gym",
            "hairsalon",
            "hospitalroom",
            "inside_bus",
            "inside_subway",
            "jewelleryshop",
            "kindergarden",
            "kitchen",
            "laboratorywet",
            "laundromat",
            "library",
            "livingroom",
            "lobby",
            "locker_room",
            "mall",
            "meeting_room",
            "movietheater",
            "museum",
            "nursery",
            "office",
            "operating_room",
            "pantry",
            "poolinside",
            "prisoncell",
            "restaurant",
            "restaurant_kitchen",
            "shoeshop",
            "stairscase",
            "studiomusic",
            "subway",
            "toystore",
            "trainstation",
            "tv_studio",
            "videostore",
            "waitingroom",
            "warehouse",
            "winecellar",
        ]

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        outcome = pd.DataFrame()
        df_size = df.size

        for i in range(df_size):
            rgb_img = df.iat[i, 0]
            pil_img = Image.fromarray(rgb_img)
            scene_prob = (
                self.model.predict(pil_img, device=self.device, verbose=False)[0]
                .probs.cpu()
                .numpy()
            )
            top_scene = self.labels[np.argmax(scene_prob)]
            label = " ".join([lb.lower() for lb in top_scene.split("_")])
            outcome = pd.concat(
                [outcome, pd.DataFrame({"labels": [label]})], ignore_index=True
            )

        return outcome
