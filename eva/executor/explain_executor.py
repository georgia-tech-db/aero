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
import pandas as pd

from eva.executor.abstract_executor import AbstractExecutor
from eva.models.storage.batch import Batch
from eva.plan_nodes.abstract_plan import AbstractPlan
from eva.plan_nodes.explain_plan import ExplainPlan
from eva.plan_nodes.function_scan_plan import FunctionScanPlan
from eva.plan_nodes.lateral_join_plan import LateralJoinPlan
from eva.plan_nodes.predicate_plan import PredicatePlan
from eva.plan_nodes.seq_scan_plan import SeqScanPlan


class ExplainExecutor(AbstractExecutor):
    def __init__(self, node: ExplainPlan):
        super().__init__(node)

    def exec(self, *args, **kwargs):
        # Traverse optimized physical plan, which is commonly supported.
        # Logical plan can be also printted by passing explainable_opr
        # attribute of the node, but is not done for now.
        plan_str = self._exec(self._node.children[0], 0)
        yield Batch(pd.DataFrame([plan_str]))

    def _exec(self, node: AbstractPlan, depth: int):
        cur_str = " " * depth * 4 + "|__ " + self._plan_str(node) + "\n"
        for child in node.children:
            cur_str += self._exec(child, depth + 1)
        return cur_str

    def _plan_str(self, plan: AbstractPlan):
        # plan_str = str(plan.__class__.__name__)
        plan_str = str(plan)
        return plan_str
