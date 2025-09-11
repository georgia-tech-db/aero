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
import os

from typing import List

import torch
import numpy as np
import pandas as pd

from PIL import Image
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

from eva.udfs.abstract.abstract_udf import AbstractClassifierUDF
from eva.utils.logging_manager import logger


class HardHatDetector(AbstractClassifierUDF):
    """
    Hard hat classifier.
    """

    def setup(self):
        pt = hf_hub_download(repo_id="keremberke/yolov8s-hard-hat-detection",
                     filename="best.pt")
        self.model = YOLO(pt)

        # Logging CUDA info.
        device_env = os.environ.get("CUDA_VISIBLE_DEVICES", str("0")) 
        logger.debug(f"CUDA-{device_env} device count: {torch.cuda.device_count()}")

        # Set device number based on env.
        self.device = int(device_env)

    @property
    def name(self) -> str:
        return "HardHatDetector"

    @property
    def labels(self) -> List[str]:
        return ["Hardhat", "NO-Hardhat"]

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        outcome = []
        df_size = df.size

        for i in range(df_size):
            rgb_img = df.iat[i, 0]
            rgb_img = rgb_img[:, :, ::-1]
            pil_img = Image.fromarray(rgb_img)
            hard_hat_boxes = self.model.predict(
                pil_img, device=self.device, verbose=False
            )[0].boxes

            labels, bboxes, scores = [], [], []

            for hard_hat_box in hard_hat_boxes:
                hard_hat_xyxy = hard_hat_box.xyxy.cpu().numpy()[0].tolist()
                hard_hat_score = hard_hat_box.conf.cpu().numpy()[0]
                hard_hat_cls = int(hard_hat_box.cls.cpu().numpy()[0])
                hard_hat_label = " ".join(
                    lb.lower() for lb in self.labels[hard_hat_cls].split(",")
                )

                labels.append(hard_hat_label)
                bboxes.append(hard_hat_xyxy)
                scores.append(hard_hat_score)

            if len(labels) > 0:
                outcome.append(
                    {
                        "labels": labels,
                        "bboxes": bboxes,
                        "scores": scores,
                    }
                )
            else:
                outcome.append(
                    {
                        "labels": ["no hardhat"],
                        "bboxes": bboxes,
                        "scores": scores,
                    }
                )

        return pd.DataFrame(
            outcome,
            columns=[
                "labels",
                "bboxes",
                "scores",
            ],
        )
