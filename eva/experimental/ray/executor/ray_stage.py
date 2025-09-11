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

from time import perf_counter, sleep
from collections import defaultdict
from threading import Thread, Event
from typing import Callable, Dict, List

import ray
from ray.exceptions import RayTaskError
from ray.util.queue import Queue, Empty

from eva.executor.executor_utils import ExecutorError
from eva.models.storage.batch import RoutingBatch, RoutingBatchStatus
from eva.utils.logging_manager import logger


class StageCompleteSignal:
    pass


@ray.remote(num_cpus=0)
def ray_stage_wait_and_alert(task_list: ray.ObjectRef, output_queue: Queue):
    try:
        ray.get(task_list)
    except RayTaskError as e:
        output_queue.put(ExecutorError(f"{e.cause}"))


@ray.remote(max_calls=1)
def ray_stage(
    executor: Callable,
    input_queue_list: List[Queue],
    output_queue_list: List[Queue],
    exch_env: Dict[str, str] = dict(),
):
    logger.debug(f"Exch - start")

    for env, value in exch_env.items():
        os.environ[env] = value
        logger.debug(f"Exch - config {env} = {value}")

    if len(input_queue_list) > 1 or len(output_queue_list) > 1:
        raise NotImplementedError

    temp_q, iq, cq = Queue(maxsize=1), input_queue_list[0], output_queue_list[0]

    # Meta-data used for statistics tracking.
    ray_worker_metrics = defaultdict(lambda: 0.0)
    monitor_st, monitor_log = perf_counter(), ""

    while True:
        routing_batch = iq.get(block=True)
        batch = routing_batch.batch
        logger.debug(
            f"Exch - get batch[{routing_batch.unique_id}] {routing_batch.debug_info}"
        )

        if batch is StageCompleteSignal:
            logger.debug(f"Exch - complete")
            break
        else:
            st = perf_counter()
            temp_q.put(batch)

            for batch in executor(input_queue_list=[temp_q]):

                # Update routing batch info.
                routing_batch.batch = batch
                routing_batch.stats_dict["EXECUTION_COST"] = (
                    perf_counter() - st
                ) * 1000
                routing_batch.status = RoutingBatchStatus.Complete
                routing_batch.debug_info = f"from Exch"

                # Update local statistics.
                ray_worker_metrics["EXECUTION_COST"] += routing_batch.stats_dict[
                    "EXECUTION_COST"
                ]

                cq.put(routing_batch)
                logger.debug(f"Exch - put batch[{routing_batch.unique_id}]")

                logger.info(
                    f"""Worker {exch_env["RAY_WORKER_ID"]} - Execution cost: {ray_worker_metrics["EXECUTION_COST"]:.3f}"""
                )

        if "LLM" in os.environ["PREDICATE_NAME"]:
            ray_worker_id = os.environ["RAY_WORKER_ID"]
            tot_exec_cost = ray_worker_metrics["EXECUTION_COST"]
            monitor_log += f"JC Balance,{ray_worker_id},{perf_counter() - monitor_st},{tot_exec_cost}\n"

    if "LLM" in os.environ["PREDICATE_NAME"]:
        ray_worker_id = os.environ["RAY_WORKER_ID"]
        with open(f"stats_worker_status_{ray_worker_id}.txt", "w") as f:
            f.write(monitor_log)
            f.flush()
