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
from typing import Callable, Dict, List

import ray
from ray.util.queue import Queue, Empty

from eva.executor.abstract_executor import AbstractExecutor
from eva.experimental.ray.executor.ray_stage import StageCompleteSignal
from eva.models.storage.batch import RoutingBatch
from eva.utils.logging_manager import logger


@ray.remote(num_cpus=1)
class BatchMonitorActor:
    def __init__(self):
        self._num_ip = 0
        self._size = 100

    def complete_batch(self):
        self._num_ip -= 1

    def add_batch(self):
        self._num_ip += 1

    def get_availability(self):
        return (self._size - self._num_ip) / self._size


@ray.remote(num_cpus=1)
def eddy_pull(
    child_executor: Callable,
    central_queue: Queue,
    eddy_pull_env: Dict[str, str] = dict(),
    batch_monitor_actor: BatchMonitorActor = None,
):
    logger.debug(f"Eddy Pull - start")

    for env, value in eddy_pull_env.items():
        os.environ[env] = value
        logger.debug(f"Eddy Pull - config {env} = {value}")

    gen = child_executor()
    unique_id = 0
    for batch in gen:
        routing_batch = RoutingBatch(
            batch=batch, unique_id=unique_id, debug_info="from Eddy Pull"
        )

        while True:
            availability_ratio = ray.get(batch_monitor_actor.get_availability.remote())
            if availability_ratio > 0.1:
                break
            # logger.debug(f"Eddy Pull - availability[{availability_ratio:.3f}]")

        central_queue.put(routing_batch)
        batch_monitor_actor.add_batch.remote()
        logger.debug(f"Eddy Pull - put batch[{routing_batch.unique_id}]")

        unique_id += 1

    # The Eddy executor has circular nature, so it is hard
    # to inject StopCompleteSignal from local function. We instead
    # voluntarily pass signal here along with total elements count.
    central_queue.put(
        RoutingBatch(
            batch=StageCompleteSignal,
            unique_id=unique_id,
        )
    )
    logger.debug(f"Eddy Pull - complete")


@ray.remote(num_cpus=1)
def eddy_route(
    router: Callable,
    central_queue: Queue,
    laminar_input_queue_list: List[Queue],
    output_queue: Queue,
    batch_monitor_actor: BatchMonitorActor = None,
):
    logger.debug(f"Eddy - start")

    while True:
        routing_batch = central_queue.get(block=True)
        batch, unique_id = routing_batch.batch, routing_batch.unique_id
        logger.debug(f"Eddy - get batch[{unique_id}] {routing_batch.debug_info}")

        # Process data.
        if batch is StageCompleteSignal:
            router.total_num_batch = unique_id
        else:
            routing_batch.debug_info = "from Eddy"
            router(routing_batch, laminar_input_queue_list, central_queue, output_queue, batch_monitor_actor)

        if router.is_stage_complete:
            break

    # Propogate complete signal to later queues.
    for input_queue in laminar_input_queue_list:
        input_queue.put(
            RoutingBatch(
                batch=StageCompleteSignal,
            )
        )
    output_queue.put(StageCompleteSignal)
    logger.debug(f"Eddy - complete")


@ray.remote(num_cpus=1)
def laminar_route(
    router: Callable, laminar_input_queue: Queue, exchange_input_queue_list: List[Queue]
):
    logger.debug(f"Laminar - start")

    while True:
        routing_batch = laminar_input_queue.get(block=True)
        batch, unique_id = routing_batch.batch, routing_batch.unique_id
        logger.debug(f"Laminar - input queue size: {laminar_input_queue.qsize()}")
        logger.debug(f"Laminar - get batch[{unique_id}] {routing_batch.debug_info}")

        if batch is StageCompleteSignal:
            break

        routing_batch.debug_info = "from Laminar"
        router(routing_batch, exchange_input_queue_list)

    # Propogate complete signal to later queues.
    for input_queue in exchange_input_queue_list:
        input_queue.put(
            RoutingBatch(
                batch=StageCompleteSignal,
            )
        )
    logger.debug("Laminar - complete")
