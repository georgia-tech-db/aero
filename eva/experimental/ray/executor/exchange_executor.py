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
from typing import Iterator

from eva.executor.abstract_executor import AbstractExecutor
from eva.executor.executor_utils import ExecutorError
from eva.experimental.ray.executor.ray_stage import (
    StageCompleteSignal,
    ray_stage,
    ray_stage_wait_and_alert,
)
from eva.experimental.ray.planner.exchange_plan import ExchangePlan
from eva.models.storage.batch import Batch
from eva.utils.logging_manager import logger


class QueueReaderExecutor(AbstractExecutor):
    def __init__(self):
        super().__init__(None)

    def exec(self, **kwargs) -> Iterator[Batch]:
        assert (
            "input_queue_list" in kwargs
        ), "Invalid ray exectuion stage. No input_queue found"

        input_queue_list = kwargs["input_queue_list"]
        assert len(input_queue_list) == 1, "Not support mulitple input queues yet."
        iq = input_queue_list[0]

        batch = iq.get(block=True)
        yield batch


class ExchangeExecutor(AbstractExecutor):
    """
    Applies predicates to filter the frames which satisfy the condition
    Arguments:
        node (AbstractPlan): The SequentialScanPlan

    """

    def __init__(self, node: ExchangePlan):
        self.parallelism = node.parallelism
        self.exch_env_list = node.exch_env_list

        # Expose a list of input queues and a unified output queue to
        # outside executor. For simplicity, all producers merge to the
        # same output queue.
        self.input_queue_list = []
        self.central_queue = None

        super().__init__(node)

    def validate(self):
        pass

    def exec(self) -> Iterator[Batch]:
        assert (
            len(self.children) == 1
        ), "Exchange executor does not support children != 1"

        # Launch Ray task based on DOP. Each input queue is assigned
        # to one producer. Now most heavylifting are shifted to the
        # Eddy operator.
        ray_task = []
        for i in range(self.parallelism):
            ray_task.append(
                ray_stage.remote(
                    self.children[0],
                    [self.input_queue_list[i]],
                    [self.central_queue],
                    self.exch_env_list[i],
                )
            )

        return ray_task

    def __call__(self, batch: Batch) -> Batch:
        pass
