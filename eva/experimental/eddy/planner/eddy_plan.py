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
from enum import Enum, auto, unique
from typing import Any, Dict, List

from eva.plan_nodes.abstract_plan import AbstractPlan
from eva.plan_nodes.types import PlanOprType


# Routing policy for eddy plan. We support two types
# of routing: 1. select one of queue; 2. select all queues
# but reorder the order to go to queue.
class RoutingPolicy(Enum):
    # Type 1.
    AllNaive = auto()  # Naive routing, no reordering.
    AllCost = auto()  # Execution cost driven routing.
    AllCostCacheAware = auto()  # Cache-ware execution cost driven routing.

    # Type Boundary.
    _BOUNDARY = auto()

    # Type 2.
    AnyRoundRobin = auto()
    AnyDynamicRoundRobin = auto()
    AnyProxyBalance = auto()


class EddyPlan(AbstractPlan):
    """
    Eddy moves data in a circular fashion. After finishing all operators,
    Eddy sends output outside.
    """

    def __init__(
        self,
        inner_plan_list: List[AbstractPlan],
        routing_policy: RoutingPolicy,
        routing_args: List[Any] = [],
        eddy_pull_env: Dict[str, str] = dict(),
    ):
        self.inner_plan_list = inner_plan_list
        self.routing_policy = routing_policy
        self.routing_args = routing_args
        self.eddy_pull_env = eddy_pull_env

        # eddy should not use Laminar routing policy
        assert (
            self.routing_policy.value < RoutingPolicy._BOUNDARY.value
        ), "EddyPlan should use All routing policy."

        super().__init__(PlanOprType.EDDY)

    def __str__(self) -> str:
        return "EddyPlan"

    def __hash__(self) -> int:
        return hash(
            (
                super().__hash__(),
                tuple(self.inner_plan_list),
                self.routing_policy,
            )
        )


class LaminarPlan(AbstractPlan):
    """
    Laminar moves data to only one operator. Typically, an operator in Laminar
    has multiple copies.
    """

    def __init__(
        self,
        inner_plan: AbstractPlan,
        routing_policy: RoutingPolicy,
        routing_args: List[Any] = [],
    ):
        self.inner_plan = inner_plan
        self.routing_policy = routing_policy
        self.routing_args = routing_args

        assert (
            self.routing_policy.value > RoutingPolicy._BOUNDARY.value
        ), "LaminarPlan should use Any routing policy."

        super().__init__(PlanOprType.LAMINAR)

    def __str__(self) -> str:
        return "LamniarPlan"

    def __hash__(self) -> int:
        return hash(
            (
                super().__hash__(),
                self.inner_plan,
                self.routing_policy,
            )
        )
