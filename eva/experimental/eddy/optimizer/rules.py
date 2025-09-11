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
from __future__ import annotations

from typing import TYPE_CHECKING

from eva.optimizer.rules.pattern import Pattern

if TYPE_CHECKING:
    from eva.optimizer.optimizer_context import OptimizerContext

from eva.configuration.configuration_manager import ConfigurationManager
from eva.experimental.eddy.planner.eddy_plan import EddyPlan, LaminarPlan, RoutingPolicy
from eva.experimental.ray.planner.exchange_plan import ExchangePlan
from eva.expression.expression_utils import to_conjunction_list
from eva.expression.function_expression import FunctionExpression
from eva.optimizer.operators import (
    LogicalApplyAndMerge,
    LogicalExchange,
    LogicalFilter,
    LogicalGet,
    LogicalProject,
    Operator,
    OperatorType,
)
from eva.optimizer.rules.rules_base import Promise, Rule, RuleType
from eva.plan_nodes.apply_and_merge_plan import ApplyAndMergePlan
from eva.plan_nodes.predicate_plan import PredicatePlan
from eva.plan_nodes.project_plan import ProjectPlan
from eva.plan_nodes.seq_scan_plan import SeqScanPlan
from eva.plan_nodes.storage_plan import StoragePlan
from eva.utils.logging_manager import logger


def str_to_policy(policy_str):
    if policy_str == "AllCost":
        return RoutingPolicy.AllCost
    elif policy_str == "AllCostCacheAware":
        return RoutingPolicy.AllCostCacheAware
    elif policy_str == "AllNaive":
        return RoutingPolicy.AllNaive
    elif policy_str == "AnyRoundRobin":
        return RoutingPolicy.AnyRoundRobin
    elif policy_str == "AnyDynamicRoundRobin":
        return RoutingPolicy.AnyDynamicRoundRobin
    elif policy_str == "AnyProxyBalance":
        return RoutingPolicy.AnyProxyBalance
    else:
        raise Exception(f"{policy_str} routing policy is not implemented")


class LogicalFilterToPhysical(Rule):
    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALFILTER)
        pattern.append_child(Pattern(OperatorType.DUMMY))
        super().__init__(RuleType.LOGICAL_FILTER_TO_PHYSICAL, pattern)

    def promise(self):
        return Promise.LOGICAL_FILTER_TO_PHYSICAL

    def check(self, grp_id: int, context: OptimizerContext):
        return True

    def apply(self, before: LogicalFilter, context: OptimizerContext):
        assert len(before.children) == 1, "Eddy can only have one child"
        lower = before.children[0]

        predicate_list = to_conjunction_list(before.predicate)
        for i, predicate in enumerate(predicate_list):
            logger.debug(f"{i}: {predicate}")

        # Configure workers and GPUs.
        config = ConfigurationManager()
        number_of_workers = config.get_value(
            "experimental", "logical_filter_to_physical_rule_workers"
        )
        number_of_gpus = config.get_value(
            "experimental", "logical_filter_to_physical_rule_gpus"
        )
        gpu_list = [
            {
                "CUDA_VISIBLE_DEVICES": str(i % number_of_gpus),
                "RAY_WORKER_ID": str(i),
            }
            # {"CUDA_VISIBLE_DEVICES": str(i // number_of_workers)}
            for i in range(number_of_workers * number_of_gpus)
        ]
        logger.debug(f"LogicalFilterToPhysical - # Config in GPU List: {len(gpu_list)}")

        # TODO:
        def is_gpu_function(pred):
            for func_name in ["DogBreedClassifier", "HardHatDetector", "YoloV5"]:
                if func_name in str(pred):
                    return True
            return False

        # Configure Laminar router.
        laminar_list = []
        laminar_routing_policy = config.get_value(
            "experimental", "laminar_routing_policy"
        )
        for predicate in predicate_list:
            logger.debug(f"Predicate: {predicate}")
            for i in range(len(gpu_list)):
                gpu_list[i]["PREDICATE_NAME"] = str(predicate)
            inner = ExchangePlan(
                parallelism=number_of_workers if is_gpu_function(predicate) else 1,
                exch_env_list=gpu_list,
            )
            inner.append_child(PredicatePlan(predicate=predicate))

            laminar_list.append(
                LaminarPlan(
                    inner_plan=inner,
                    routing_policy=str_to_policy(laminar_routing_policy),
                    routing_args=[str(predicate), number_of_gpus],
                )
            )

        # Configure Eddy router.
        eddy_routing_policy = config.get_value("experimental", "eddy_routing_policy")
        eddy_routing_args = []
        if str_to_policy(eddy_routing_policy) == RoutingPolicy.AllCost:
            eddy_routing_args.append(
                config.get_value("experimental", "eddy_ranking_function")
            )
        elif str_to_policy(eddy_routing_policy) == RoutingPolicy.AllCostCacheAware:
            eddy_routing_args.append(predicate_list)
        eddy = EddyPlan(
            inner_plan_list=laminar_list,
            routing_policy=str_to_policy(eddy_routing_policy),
            routing_args=eddy_routing_args,
            eddy_pull_env={"CUDA_VISIBLE_DEVICES": "0"},
        )
        eddy.append_child(lower)
        yield eddy


class LogicalGetToSeqScan(Rule):
    def __init__(self):
        pattern = Pattern(OperatorType.LOGICALGET)
        super().__init__(RuleType.LOGICAL_GET_TO_SEQSCAN, pattern)

    def promise(self):
        return Promise.LOGICAL_GET_TO_SEQSCAN

    def check(self, before: Operator, context: OptimizerContext):
        return True

    def apply(self, before: LogicalGet, context: OptimizerContext):
        # Configure the batch_mem_size. It decides the number of rows
        # read in a batch from storage engine.
        # ToDO: Experiment heuristics.

        batch_mem_size = 30000000  # 30mb
        config_batch_mem_size = ConfigurationManager().get_value(
            "executor", "batch_mem_size"
        )
        if config_batch_mem_size:
            batch_mem_size = config_batch_mem_size
        scan = SeqScanPlan(None, before.target_list, before.alias)
        lower = StoragePlan(
            before.table_obj,
            batch_mem_size=batch_mem_size,
            predicate=before.predicate,
            sampling_rate=before.sampling_rate,
        )

        # Check whether the projection contains a UDF
        if before.target_list is None or not any(
            [isinstance(expr, FunctionExpression) for expr in before.target_list]
        ):
            # Reset number of consumer to 1, because the SeqScan above it will
            # not be replicated.
            scan.append_child(lower)
            yield scan
        else:
            # Creates Eddy Plan with single laminar just to have multiple workers

            # Hardcoded GPU configuration.
            number_of_workers = ConfigurationManager().get_value(
                "experimental", "logical_get_to_sequential_scan_workers"
            )
            number_of_gpus = ConfigurationManager().get_value(
                "experimental", "logical_get_to_sequential_scan_gpus"
            )
            gpu_list = [{"CUDA_VISIBLE_DEVICES": i} for i in range(number_of_gpus)]

            inner = ExchangePlan(parallelism=number_of_workers, exch_env=gpu_list)
            inner.append_child(scan)

            laminar = LaminarPlan(
                inner_plan=inner, routing_policy=RoutingPolicy.AnyRoundRobin
            )
            eddy = EddyPlan(
                inner_plan_list=[laminar],
                routing_policy=RoutingPolicy.AllNaive,
            )

            # The Storage plan is still child of eddy.
            eddy.append_child(lower)

            yield eddy
