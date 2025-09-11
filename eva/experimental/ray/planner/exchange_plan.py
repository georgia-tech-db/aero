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

from typing import Dict, List

from eva.plan_nodes.abstract_plan import AbstractPlan
from eva.plan_nodes.types import PlanOprType


class ExchangePlan(AbstractPlan):
    def __init__(self, parallelism: int = 1, exch_env_list: List[Dict[str, str]] = [dict()]):
        self.parallelism = parallelism
        self.exch_env_list = exch_env_list
        super().__init__(PlanOprType.EXCHANGE)

    def __str__(self) -> str:
        return "ExchangePlan"

    def __hash__(self) -> int:
        hash_exch_env_list = []
        for exch_env in self.exch_env_list:
            for k, v in exch_env.items():
                hash_exch_env_list.append((k, v))
        return hash(
            (
                super().__hash__(), 
                self.parallelism, 
                frozenset(hash_exch_env_list)
            )
        )
