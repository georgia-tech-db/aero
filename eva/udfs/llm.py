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

from gpt4all import GPT4All

from eva.executor.executor_utils import ExecutorError
from eva.udfs.abstract.abstract_udf import AbstractUDF
from eva.utils.logging_manager import logger


class LLM(AbstractUDF):
    def setup(self):
        self.model = GPT4All("orca-2-13b.Q4_0.gguf", n_threads=64, verbose=False)

    @property
    def name(self):
        return "LLM"

    def forward(self, df: pd.DataFrame) -> pd.DataFrame:
        # logger.warning(f"LLM internal: {df}")

        def _llm(row: pd.Series) -> str:
            row = row.to_list()
            prompt, question = row[0], row[1]
            input_text = f"{prompt}\n{question}\n"
            response = self.model.generate(input_text, max_tokens=300)
            response_arr = response.split("'")
            if len(response_arr) < 3:
                if "service" in prompt:
                    response = "service"
                else:
                    response = "negative"
            else:
                response = response_arr[1]
            # logger.warning(f"Prompt: {prompt}\nQuestion: {question}\nResponse: {response}")
            return response

        ret = pd.DataFrame()
        ret["response"] = df.apply(_llm, axis=1)
        return ret