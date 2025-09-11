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
import math
import pandas as pd

from abc import ABC
from collections import defaultdict
from typing import List

from ray.util.queue import Queue

from eva.configuration.configuration_manager import ConfigurationManager
from eva.experimental.eddy.planner.eddy_plan import RoutingPolicy
from eva.experimental.eddy.util.profiler import Profiler
from eva.experimental.eddy.executor.eddy_remote import BatchMonitorActor
from eva.models.storage.batch import RoutingBatch, Batch, RoutingBatchStatus
from eva.utils.logging_manager import logger


def create_router(routing_policy, routing_args):
    router = None
    if routing_policy == RoutingPolicy.AnyRoundRobin:
        router = AnyRoundRobin(*routing_args)
    elif routing_policy == RoutingPolicy.AnyDynamicRoundRobin:
        router = AnyDynamicRoundRobin(*routing_args)
    elif routing_policy == RoutingPolicy.AnyProxyBalance:
        router = AnyProxyBalance(*routing_args)
    elif routing_policy == RoutingPolicy.AllNaive:
        router = AllNaive(*routing_args)
    elif routing_policy == RoutingPolicy.AllCost:
        router = AllCost(*routing_args)
    elif routing_policy == RoutingPolicy.AllCostCacheAware:
        router = AllCostCacheAware(*routing_args)
    else:
        raise NotImplementedError(
            "{} routing policy is not implemented.".format(routing_policy)
        )

    return router


class AbstractRouter(ABC):
    def __call__(self, tuple, input_queue_list):
        raise NotImplementedError


################################################################
# Any routing algorithm for laminar.
################################################################


def start_all_profiling(profiler_list):
    for prof in profiler_list:
        prof.start()


def stop_all_profiling(profiler_list):
    for prof in profiler_list:
        prof.stop()


def reset_all_profiling(profiler_list):
    for prof in profiler_list:
        prof.reset()


def print_gpu_util(name, profiler_list, logger):
    util_str = ""
    for prof in profiler_list:
        util_str += " GPU act util: {:.3f}, GPU temp util: {:.3f}, GPU mem usage: {:.3f}".format(
            prof.actual_gpu_util,
            prof.temporal_gpu_util,
            prof.memory_usage,
        )
    logger.debug(f"Laminar {name} Route (AdaptiveScalingRoundRobin) - {util_str}")


class AnyProxyBalance(AbstractRouter):
    def __init__(self, name, number_of_gpus):
        self._idx = 0
        self._name = name
        self._number_of_gpus = number_of_gpus

        # ! Policy mandate.
        # Hacky way to bypass routing policy on simple predicate.
        self._is_bypass = "()" not in self._name

        # A proxy cost list.
        self._cost_list = None

    def __call__(
        self,
        routing_batch,
        input_queue_list,
    ):
        if self._cost_list is None:
            self._cost_list = [0 for _ in range(len(input_queue_list))]
            self._idx = 0

        if not self._is_bypass:
            # Choose a better cost queue to schedule.
            schedule_cost = 0xffffffff
            for idx, cost in enumerate(self._cost_list):
                if cost < schedule_cost:
                    schedule_cost = cost
                    self._idx = idx
        else:
            self._idx = 0

        routing_batch.debug_info = f"from Laminar {self._name} (AnyProxyBalance)"
        input_queue_list[self._idx].put(routing_batch)

        # Update cost statistics.
        proxy_column, df = None, routing_batch.batch.frames
        for column in df.columns:
            if "review" in column:
                proxy_column = column
        df["str_length"] = df[proxy_column].str.len()
        proxy_cost = len(df)
        self._cost_list[self._idx] += proxy_cost

        logger.debug(
            f"Lamiar {self._name} Route (AnyProxyBalance - Cost: {self._cost_list})"
        )
        logger.debug(
            f"Laminar {self._name} Route (AnyProxyBalance) - batch[{routing_batch.unique_id}] routes to IQ[{self._idx}]"
        )

        self._idx = (self._idx + 1) % len(input_queue_list)


class AnyRoundRobin(AbstractRouter):
    def __init__(self, name, number_of_gpus):
        self._idx = 0
        self._name = name
        self._number_of_gpus = number_of_gpus
        self._profiler_list = None

        # ! Policy mandate.
        # Hacky way to bypass routing policy on simple predicate.
        self._is_bypass = "()" not in self._name

        # A proxy cost list.
        self._cost_list = None

    def __call__(
        self,
        routing_batch,
        input_queue_list,
    ):
        if self._profiler_list is None:
            self._profiler_list = [
                Profiler(gpu_idx=i) for i in range(self._number_of_gpus)
            ]

        if self._cost_list is None:
            self._cost_list = [0 for _ in range(len(input_queue_list))]

        if self._is_bypass:
            self._idx = 0
        # else:
        #     self._idx = routing_batch.unique_id % len(input_queue_list)

        routing_batch.debug_info = f"from Laminar {self._name} (AnyRoundRobin)"
        input_queue_list[self._idx].put(routing_batch)

        # Update cost statistics.
        df = routing_batch.batch.frames
        proxy_cost = len(df)
        self._cost_list[self._idx] += proxy_cost

        logger.debug(
            f"Lamiar {self._name} Route (AnyRoundRobin) - Cost: {self._cost_list}"
        )
        logger.debug(
            f"Laminar {self._name} Route (AnyRoundRobin) - batch[{routing_batch.unique_id}] routes to IQ[{self._idx}]"
        )

        self._idx = (self._idx + 1) % len(input_queue_list)


class AnyDynamicRoundRobin(AbstractRouter):
    def __init__(self, name, number_of_gpus):
        self._idx = 0
        self._name = name
        self._number_of_gpus = number_of_gpus
        self._profiler_list = None

        # Max number of queues.
        self._max_number_of_queues = 0xFFFFFFFF

    def __call__(
        self,
        routing_batch,
        input_queue_list,
    ):
        if self._profiler_list is None:
            self._profiler_list = [
                Profiler(gpu_idx=i) for i in range(self._number_of_gpus)
            ]

        routing_batch.debug_info = f"from Laminar {self._name} (AnyDynamicRoundRobin)"

        # Adjust routing index based on whether it is warm up.
        if (
            not routing_batch.flags_dict["IS_WARM_UP"]
            and self._max_number_of_queues == 0xFFFFFFFF
        ):
            # Use GPU 0 as the guideline for choosing the parallelism.
            prof = self._profiler_list[0]
            self._max_number_of_queues = (
                int(1 / prof.memory_usage) * self._number_of_gpus
            )

            logger.debug(
                f"Laminar {self._name} Route (AnyDynamicRoundRobin) - GPUs: {self._number_of_gpus} memory usage: {prof.memory_usage:.3f}"
            )
            logger.debug(
                f"Laminar {self._name} Route (AnyDynamicRoundRobin) - max queues update to {self._max_number_of_queues}"
            )

            if self._idx >= self._max_number_of_queues:
                self._idx = 0

        # Actual routing.
        input_queue_list[self._idx].put(routing_batch)
        logger.debug(
            f"Laminar {self._name} Route (AnyDynamicRoundRobin) - batch[{routing_batch.unique_id}] routes to IQ[{self._idx}]"
        )

        self._idx = (self._idx + 1) % min(
            len(input_queue_list), self._max_number_of_queues
        )


################################################################
# All routing algorithm for eddy.
################################################################


class GenericEddyRouter(AbstractRouter):
    def __init__(self):
        self._complete_num_batch = 0
        self.total_num_batch = None

        # Data structure to track which predicate has been visited.
        self._visited_predicate = defaultdict(lambda: set([]))

    @property
    def router_name(self):
        raise NotImplementedError

    @property
    def is_stage_complete(self):
        if self.total_num_batch:
            if self._complete_num_batch == self.total_num_batch:
                return True
        return False

    def clear_all_flags(self, routing_batch):
        for k, _ in routing_batch.flags_dict.items():
            routing_batch.flags_dict[k] = False

    def routing_batch_has_visited_all(self, routing_batch, input_queue_list):
        for i, _ in enumerate(input_queue_list):
            if (
                i not in self._visited_predicate
                or routing_batch.unique_id not in self._visited_predicate[i]
            ):
                return False
        return True

    def routing_batch_visit_input_queue_at_idx(self, routing_batch, idx):
        self._visited_predicate[idx].add(routing_batch.unique_id)

    def get_routing_batch_debug_info(self):
        return f"from Eddy ({self.router_name})"

    def output_route(
        self,
        routing_batch: RoutingBatch,
        output_queue: Queue,
        batch_monitor_actor: BatchMonitorActor = None,
    ):
        """
        Route to output queue.
        """
        self._complete_num_batch += 1
        output_queue.put(routing_batch.batch)
        batch_monitor_actor.complete_batch.remote()
        logger.debug(
            f"Eddy ({self.router_name}) - batch[{routing_batch.unique_id}] complete. "
            + f"Total: {self.total_num_batch}, complete: {self._complete_num_batch}."
        )

    def circular_route(
        self,
        routing_batch: RoutingBatch,
        central_queue: Queue,
    ):
        """
        Route batch back to the central queue.
        """
        central_queue.put(routing_batch)

        logger.debug(
            f"Eddy ({self.router_name}) - (Circular) batch[{routing_batch.unique_id}] routes"
        )


class AllNaive(GenericEddyRouter):
    @property
    def router_name(self):
        return "AllNaive"

    def __call__(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
        central_queue: Queue,
        output_queue: Queue,
        batch_monitor_actor: BatchMonitorActor = None,
    ):
        # Mark complete for a queue.
        if routing_batch.status == RoutingBatchStatus.Complete:
            self.routing_batch_visit_input_queue_at_idx(
                routing_batch, routing_batch.input_queue_index
            )

        if not routing_batch.batch.empty():
            for i, input_queue in enumerate(input_queue_list):
                if routing_batch.unique_id not in self._visited_predicate[i]:
                    # Prepare per routing batch statistics.
                    routing_batch.debug_info = self.get_routing_batch_debug_info()
                    routing_batch["INPUT_QUEUE_INDEX"] = i

                    # Actual routing.
                    input_queue.put(routing_batch)

                    logger.debug(
                        f"Eddy (AllNaive) - batch[{routing_batch.unique_id}] routes to IQ[{i}]"
                    )
                    return

        self.output_route(routing_batch, output_queue, batch_monitor_actor)


def logger_info_prioritized_queue(
    routing_batch, visited_predicate, sorted_input_queue_list
):
    not_all = True
    for _, visited in visited_predicate.items():
        if routing_batch.unique_id in visited:
            not_all = False
    if not_all:
        print_input_queue_list = sorted(
            sorted_input_queue_list, key=lambda item: item[0]
        )
        cost_str = ",".join([str(cost) for _, cost in print_input_queue_list])
        logger.info(
            f" jcplan {routing_batch.batch.frames['warehousevideo.id'][0]},{cost_str} jcplan "
        )


class AllCost(GenericEddyRouter):
    @property
    def router_name(self):
        return "AllCost"

    def __init__(self, ranking_function):
        super().__init__()

        # Input queue cost.
        self._call_time = 0
        self._input_queue_cost = defaultdict(lambda: 0.0)
        self._input_queue_batch_size = defaultdict(lambda: 0)
        self._input_queue_filter_batch_size = defaultdict(lambda: 0)

        # Whether router is at warm-up stage.
        self._is_warmup = True

        # Ranking function for cost routing.
        self._ranking_function = ranking_function

    def _get_cost(self, input_queue_index):
        assert self._input_queue_batch_size[input_queue_index] != 0
        avg_cost = (
            self._input_queue_cost[input_queue_index]
            / self._input_queue_batch_size[input_queue_index]
        )
        avg_sel = (
            1
            - self._input_queue_filter_batch_size[input_queue_index]
            / self._input_queue_batch_size[input_queue_index]
        )
        if self._ranking_function == "Cost":
            return avg_cost
        elif self._ranking_function == "Selectivity":
            return avg_sel
        elif self._ranking_function == "Score":
            return avg_cost / (1 - avg_sel)
        else:
            raise Exception(
                f"{self._ranking_function} ranking function is not implemented"
            )

    def _get_sorted_input_queue_list_based_profile_execution_cost(
        self,
        input_queue_list,
    ):
        sorted_input_queue_list = [
            (i, self._get_cost(i)) for i in range(len(input_queue_list))
        ]
        # sorted_input_queue_list[0] = (0, 0)  # Enforce the order of = 'dog' predicate
        sorted_input_queue_list = sorted(sorted_input_queue_list, key=lambda x: x[1])
        return sorted_input_queue_list

    def _update_queue_statistics(self, routing_batch):
        input_queue_idx = routing_batch.stats_dict["INPUT_QUEUE_INDEX"]
        self.routing_batch_visit_input_queue_at_idx(
            routing_batch, routing_batch.stats_dict["INPUT_QUEUE_INDEX"]
        )
        self._input_queue_cost[input_queue_idx] += routing_batch.stats_dict[
            "EXECUTION_COST"
        ]
        self._input_queue_batch_size[input_queue_idx] += routing_batch.stats_dict[
            "INPUT_QUEUE_BATCH_SIZE"
        ]
        self._input_queue_filter_batch_size[
            input_queue_idx
        ] += routing_batch.stats_dict["INPUT_QUEUE_BATCH_SIZE"] - len(
            routing_batch.batch
        )

    def _log_queue_statistics(self, routing_batch, input_queue_list):
        log_str = ""
        for i, _ in enumerate(input_queue_list):
            if i in self._input_queue_batch_size:
                log_str += (
                    f"Eddy ({self.router_name}) - batch[{routing_batch.unique_id}] - IQ[{i}] - "
                    + f"Avg Cost: {self._input_queue_cost[i] / self._input_queue_batch_size[i]:.3f}"
                    + f", Avg Sel: {1 - self._input_queue_filter_batch_size[i] / self._input_queue_batch_size[i]:.3f}"
                    + f", Usage: {self._input_queue_batch_size[i]}\n"
                )
        logger.debug(log_str)

    def __call__(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
        central_queue: Queue,
        output_queue: Queue,
        batch_monitor_actor: BatchMonitorActor = None,
    ):
        # Initial statistics update.
        if routing_batch.status == RoutingBatchStatus.Complete:
            self._update_queue_statistics(routing_batch)

        # Log statistics.
        self._log_queue_statistics(routing_batch, input_queue_list)

        # Clear all flags.
        self.clear_all_flags(routing_batch)

        # Mark routing batch status again.
        routing_batch.status = RoutingBatchStatus.InProgress

        # Check if execution cost of all predicates are ready.
        stats_ready = len(self._input_queue_batch_size) == len(input_queue_list)

        if routing_batch.batch.empty() or self.routing_batch_has_visited_all(
            routing_batch, input_queue_list
        ):
            self.output_route(routing_batch, output_queue, batch_monitor_actor)
        elif not self._is_warmup and not stats_ready:
            self.circular_route(routing_batch, central_queue)
        else:
            if self._is_warmup and routing_batch.unique_id >= self._call_time:
                self.warmup_route(routing_batch, input_queue_list)
                self._call_time += 1
                if self._call_time == len(input_queue_list):
                    self._is_warmup = False
            elif self._is_warmup:
                self.circular_route(routing_batch, central_queue)
            else:
                self.laminar_route(routing_batch, input_queue_list)


    def warmup_route(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
    ):
        """
        Route batch in round-robin fashion to get statistics as warm up.
        """

        # Statistics update.
        routing_batch.debug_info = self.get_routing_batch_debug_info()
        routing_batch.stats_dict["INPUT_QUEUE_INDEX"] = self._call_time
        routing_batch.stats_dict["INPUT_QUEUE_BATCH_SIZE"] = len(routing_batch.batch)
        routing_batch.flags_dict["IS_WARM_UP"] = True

        # Actual routing.
        input_queue_list[self._call_time].put(routing_batch)

        logger.debug(
            f"Eddy ({self.router_name}) - (Warmup) batch[{routing_batch.unique_id}] routes to IQ[{self._call_time}]"
        )

    def laminar_route(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
    ):
        """
        Route to laminar queues according to statistics.
        """

        # Sort the routing order based on cost. Initially, when no known cost is avaiable yet,
        # use queue length as cost.
        sorted_input_queue_list = (
            self._get_sorted_input_queue_list_based_profile_execution_cost(
                input_queue_list,
            )
        )
        logger.debug(
            f"Eddy (AllCost) - batch[{routing_batch.unique_id}] sorted queue: "
            + f"{', '.join([str(i) + ':' + str(round(cost, 3)) for i, cost in sorted_input_queue_list])}"
        )

        # Routing.
        for queue_idx, (i, _) in enumerate(sorted_input_queue_list):
            if routing_batch.unique_id not in self._visited_predicate[i]:
                # Print prioritized queue
                # if queue_idx == 0:
                #     logger_info_prioritized_queue(
                #         routing_batch, self._visited_predicate, sorted_input_queue_list
                #     )

                # Update statistics.
                routing_batch.debug_info = self.get_routing_batch_debug_info()
                routing_batch.stats_dict["INPUT_QUEUE_INDEX"] = i
                routing_batch.stats_dict["INPUT_QUEUE_BATCH_SIZE"] = len(
                    routing_batch.batch
                )

                # Actual routing.
                input_queue_list[i].put(routing_batch)

                logger.debug(
                    f"Eddy (AllCost) - batch[{routing_batch.unique_id}] routes to IQ[{i}]"
                )
                return


class AllCostCacheAware(GenericEddyRouter):
    @property
    def router_name(self):
        return "AllCostCacheAware"

    def __init__(self, predicate_list):
        super().__init__()

        # Input queue cost.
        self._call_time = 0
        self._input_queue_cost = defaultdict(lambda: 0.0)
        self._input_queue_batch_size = defaultdict(lambda: 0)
        self._input_queue_filter_batch_size = defaultdict(lambda: 0)
        self._input_queue_uncached_batch_size = defaultdict(lambda: 0)

        # Used to check cache.
        self._predicate_list = predicate_list

        # Whether router is at warm-up stage.
        self._is_warmup = True

    def _update_queue_statistics(self, routing_batch):
        input_queue_idx = routing_batch.stats_dict["INPUT_QUEUE_INDEX"]
        self.routing_batch_visit_input_queue_at_idx(
            routing_batch, routing_batch.stats_dict["INPUT_QUEUE_INDEX"]
        )
        self._input_queue_cost[input_queue_idx] += routing_batch.stats_dict[
            "EXECUTION_COST"
        ]
        self._input_queue_batch_size[input_queue_idx] += routing_batch.stats_dict[
            "INPUT_QUEUE_BATCH_SIZE"
        ]
        self._input_queue_filter_batch_size[
            input_queue_idx
        ] += routing_batch.stats_dict["INPUT_QUEUE_BATCH_SIZE"] - len(
            routing_batch.batch
        )
        self._input_queue_uncached_batch_size[
            input_queue_idx
        ] += routing_batch.stats_dict["INPUT_QUEUE_UNCACHED_BATCH_SIZE"]

    def _log_queue_statistics(self, routing_batch, input_queue_list):
        for i, _ in enumerate(input_queue_list):
            if i in self._input_queue_batch_size:
                log_str = (
                    f"Eddy (AllCostCacheAware) - batch[{routing_batch.unique_id}] - IQ[{i}] - "
                    + f"Avg Cost: {self._get_cost(i) * 1000:.3f}"
                    + f", Avg Sel: {1 - self._input_queue_filter_batch_size[i] / self._input_queue_batch_size[i]:.3f}"
                    + f", Usage: {self._input_queue_batch_size[i]}"
                    + f", Uncached Usage: {self._input_queue_uncached_batch_size[i]}"
                )
                logger.debug(log_str)

    def _get_cost(self, input_queue_index):
        if self._input_queue_uncached_batch_size[input_queue_index] == 0:
            return 0
        else:
            return (
                self._input_queue_cost[input_queue_index]
                / self._input_queue_uncached_batch_size[input_queue_index]
            )

    def _get_sorted_input_queue_list_based_profile_execution_cost(
        self,
        input_queue_list,
        uncached_batch_size_list,
    ):
        sorted_input_queue_list = [
            (i, self._get_cost(i) * uncached_batch_size_list[i])
            for i in range(len(input_queue_list))
        ]
        sorted_input_queue_list = sorted(sorted_input_queue_list, key=lambda x: x[1])
        return sorted_input_queue_list

    def __call__(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
        central_queue: Queue,
        output_queue: Queue,
        batch_monitor_actor: BatchMonitorActor = None,
    ):
        # Initial statistics update.
        if routing_batch.status == RoutingBatchStatus.Complete:
            self._update_queue_statistics(routing_batch)

        # Log statistics.
        self._log_queue_statistics(routing_batch, input_queue_list)

        # Check if execution cost of all predicates are ready.
        stats_ready = len(self._input_queue_batch_size) == len(input_queue_list)

        if routing_batch.batch.empty() or self.routing_batch_has_visited_all(
            routing_batch, input_queue_list
        ):
            self.output_route(routing_batch, output_queue, batch_monitor_actor)
        elif not self._is_warmup and not stats_ready:
            self.circular_route(routing_batch, central_queue)
        else:
            # Calculate usage which is based on both # of uncached rows and total # of rows.
            # usage = len(batch) * uncached_ratio.
            uncached_batch_size_list = [
                predicate.get_uncached_ratio(routing_batch.batch)
                * len(routing_batch.batch)
                for predicate in self._predicate_list
            ]
            logger.debug(
                f"Eddy ({self.router_name}) - batch[{routing_batch.unique_id}] uncached batch size "
                + f"{', '.join([str(int(size)) for size in uncached_batch_size_list])}"
            )

            if self._is_warmup and routing_batch.unique_id >= self._call_time:
                self.warmup_route(
                    routing_batch, input_queue_list, uncached_batch_size_list
                )
                self._call_time += 1
                if self._call_time == len(input_queue_list):
                    self._is_warmup = False
            elif self._is_warmup:
                self.circular_route(routing_batch, central_queue)
            else:
                self.laminar_route(
                    routing_batch, input_queue_list, uncached_batch_size_list
                )

    def warmup_route(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
        uncached_batch_size_list: List[float],
    ):
        """
        Route batch in round-robin fashion to get statistics as warm up.
        """

        # Update statistics.
        routing_batch.debug_info = self.get_routing_batch_debug_info()
        routing_batch.stats_dict["INPUT_QUEUE_INDEX"] = self._call_time
        routing_batch.stats_dict["INPUT_QUEUE_BATCH_SIZE"] = len(routing_batch.batch)
        routing_batch.stats_dict[
            "INPUT_QUEUE_UNCACHED_BATCH_SIZE"
        ] = uncached_batch_size_list[self._call_time]

        # Actual routing.
        input_queue_list[self._call_time].put(routing_batch)

        logger.debug(
            f"Eddy (AllCostCacheAware) - (Warmup) batch[{routing_batch.unique_id}] routes to IQ[{self._call_time}]"
        )

    def laminar_route(
        self,
        routing_batch: RoutingBatch,
        input_queue_list: List[Queue],
        uncached_batch_size_list: List[float],
    ):
        """
        Route to laminar queues according to statistics.
        """

        # Sort the routing order based on cost. Initially, when no known cost is avaiable yet,
        # use queue length as cost.
        sorted_input_queue_list = (
            self._get_sorted_input_queue_list_based_profile_execution_cost(
                input_queue_list,
                uncached_batch_size_list,
            )
        )
        logger.debug(
            f"Eddy (AllCostCacheAware) - batch[{routing_batch.unique_id}] sorted queue: "
            + f"{', '.join([str(i) + ':' + str(round(cost, 3)) for i, cost in sorted_input_queue_list])}"
        )

        # Routing.
        for queue_idx, (i, _) in enumerate(sorted_input_queue_list):
            if routing_batch.unique_id not in self._visited_predicate[i]:
                # Print prioritized queue
                # if queue_idx == 0:
                #     logger_info_prioritized_queue(
                #         routing_batch, self._visited_predicate, sorted_input_queue_list
                #     )

                # Update statistics.
                routing_batch.debug_info = self.get_routing_batch_debug_info()
                routing_batch.stats_dict["INPUT_QUEUE_INDEX"] = i
                routing_batch.stats_dict["INPUT_QUEUE_BATCH_SIZE"] = len(
                    routing_batch.batch
                )
                routing_batch.stats_dict[
                    "INPUT_QUEUE_UNCACHED_BATCH_SIZE"
                ] = uncached_batch_size_list[i]

                # Actual routing.
                input_queue_list[i].put(routing_batch)

                logger.debug(
                    f"Eddy (AllCostCacheAware) - batch[{routing_batch.unique_id}] routes to IQ[{i}]"
                )
                return
