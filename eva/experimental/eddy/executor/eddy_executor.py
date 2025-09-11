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

from ray.util.queue import Queue

from eva.executor.abstract_executor import AbstractExecutor
from eva.executor.executor_utils import ExecutorError
from eva.experimental.eddy.executor.eddy_remote import (
    eddy_pull,
    eddy_route,
    laminar_route,
    BatchMonitorActor,
)
from eva.experimental.eddy.executor.router import create_router
from eva.experimental.eddy.planner.eddy_plan import EddyPlan, LaminarPlan
from eva.experimental.ray.executor.exchange_executor import QueueReaderExecutor
from eva.experimental.ray.executor.ray_stage import (
    StageCompleteSignal,
    ray_stage_wait_and_alert,
)
from eva.experimental.ray.planner.exchange_plan import ExchangePlan
from eva.models.storage.batch import Batch
from eva.utils.logging_manager import logger


def traverse_leaf_executor(executor):
    assert (
        len(executor.children) <= 1
    ), "Queue logic currently does not support multiple inputs."

    if len(executor.children) > 0:
        return traverse_leaf_executor(executor.children[0])
    return executor


"""
Design of Eddy and Laminar executors.

Link: https://docs.google.com/presentation/d/1GYxPRGur6wxq-j41TCwkYCh-YaLQPuzv4tfU3UnoJr0/edit?usp=sharing
"""


class LaminarExecutor(AbstractExecutor):
    def __init__(self, node: LaminarPlan):
        self.inner_plan = node.inner_plan
        self.routing_policy = node.routing_policy
        self.routing_args = node.routing_args

        assert isinstance(
            self.inner_plan, ExchangePlan
        ), "LaminarExecutor inner executor only takes ExchangeExecutor."

        # Construct the routing queues.
        self.exchange_input_queue_list = []
        logger.debug(f"LaminarExecutor - # of parallelism: {self.inner_plan.parallelism}")
        for _ in range(self.inner_plan.parallelism):
            self.exchange_input_queue_list.append(Queue(maxsize=2))

        # The central queue and Laminar input queue will be populated
        # from the Eddy executor.
        self.central_queue = None
        self.laminar_input_queue = None

        # Storing the inner execution tree. It will be populated
        # during construction of the execution tree.
        self.inner_executor = None

        # Routing callable.
        self.router = create_router(self.routing_policy, self.routing_args)

        super().__init__(node)

    def validate(self):
        pass

    def exec(self) -> Iterator[Batch]:
        logger.debug(f"LaminarExecutor - start")

        # Start Exchange executor.
        ray_task = self.inner_executor.exec()

        # Laminar routing pulls data from input queue to Laminar and
        # routes data to input queues to Exchange.
        ray_task += [
            laminar_route.remote(
                self.router,
                self.laminar_input_queue,
                self.exchange_input_queue_list,
            )
        ]

        return ray_task

    # This function accepts the execution tree built outside and
    # it attaches appropriate queue to each operator. Last, it
    # also appends a queue reader operator at each operator's
    # leaf level.
    def build_inner_executor(self, inner_executor):
        self.inner_executor = inner_executor
        leaf_executor = traverse_leaf_executor(self.inner_executor)
        queue_reader_executor = QueueReaderExecutor()
        leaf_executor.children.append(queue_reader_executor)

    def populate_input_queue_list(self):
        self.inner_executor.input_queue_list = self.exchange_input_queue_list
        self.inner_executor.central_queue = self.central_queue

    def __call__(self, batch: Batch) -> Batch:
        pass

    def __str__(self):
        return "LaminarExecutor"


class EddyExecutor(AbstractExecutor):
    def __init__(self, node: EddyPlan):
        self.inner_plan_list = node.inner_plan_list
        self.routing_policy = node.routing_policy
        self.routing_args = node.routing_args
        self.eddy_pull_env = node.eddy_pull_env

        for plan in self.inner_plan_list:
            assert isinstance(
                plan, LaminarPlan
            ), "EddyOperator should only be connected to FlowOperator."

        # Input and output queues for inner list of plans.
        self.laminar_input_queue_list = []
        for plan in self.inner_plan_list:
            logger.debug(f"EddyExecutor - plan {plan}")
            self.laminar_input_queue_list.append(Queue(maxsize=2))

        # Central queue to route all data.
        self.central_queue = Queue(maxsize=100)

        # Final output queue to collect results.
        self.output_queue = Queue(maxsize=100)

        # Storing the inner execution tree. It will be populated
        # during construction of the execution tree.
        self.inner_executor_list = []

        # Routing callable.
        self.router = create_router(self.routing_policy, self.routing_args)

        super().__init__(node)

    def validate(self):
        pass

    def exec(self) -> Iterator[Batch]:
        logger.debug(f"EddyExecutor - start")

        # Start with sanity check.
        assert (
            len(self.children) == 1
        ), "EddyOperator currently does not support children != 1."

        ray_task = []

        # Start inner Laminar operator.
        for executor in self.inner_executor_list:
            ray_task += executor.exec()

        # Launch task to route data from central queue to
        # input queue or output queue.
        batch_monitor_actor = BatchMonitorActor.remote()

        ray_task += [
            eddy_route.remote(
                self.router,
                self.central_queue,
                self.laminar_input_queue_list,
                self.output_queue,
                batch_monitor_actor,
            )
        ]

        # Launch task to pull from children operators.
        ray_task += [
            eddy_pull.remote(
                self.children[0],
                self.central_queue,
                self.eddy_pull_env,
                batch_monitor_actor,
            )
        ]

        # Yield from the output queue. In this case, remote producers are
        # asynchronously writing to the queue.
        ray_stage_wait_and_alert.remote(ray_task, self.output_queue)
        while True:
            res = self.output_queue.get(block=True)
            if res is StageCompleteSignal:
                break
            elif isinstance(res, ExecutorError):
                raise res
            else:
                yield res

    # This function accepts the execution tree built outside and
    # it attaches appropriate queue to each operator.
    def build_inner_executor_list(self, inner_executor_list):
        self.inner_executor_list = inner_executor_list

    def populate_input_queue_list(self):
        for i, executor in enumerate(self.inner_executor_list):
            executor.central_queue = self.central_queue
            executor.laminar_input_queue = self.laminar_input_queue_list[i]

    def __call__(self, batch: Batch) -> Batch:
        pass

    def __str__(self):
        return "EddyExecutor"
