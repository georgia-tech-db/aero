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

import pytest
import ray

from eva.configuration.configuration_manager import ConfigurationManager
from eva.executor.execution_context import Context
from eva.server.command_handler import execute_query_fetch_all
from eva.utils.stats import Timer


USE_CASE_4_QUERY = f"""SELECT * FROM FoodReview
        WHERE LLM("What is the following review about? Choose only one: 'service' or 'food'", review) = "food"
        AND rating <= 1;
    """


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_4_baseline(benchmark, setup_pytorch_tests):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "eddy", False)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 1)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)

    # Actual query execution.
    timer = Timer()
    with timer:
        actual_batch = execute_query_fetch_all(USE_CASE_4_QUERY)
    print("Query time", timer.total_elapsed_time)
    print(len(actual_batch))


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_4_eddies(benchmark, setup_pytorch_tests):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 2)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Cost")
    config.update_value("experimental", "laminar_routing_policy", "AnyRoundRobin")

    context = Context()
    ray.init(num_gpus=len(context.gpus))

    # Actual query execution.
    config.update_value("experimental", "eddy", True)
    timer = Timer()
    with timer:
        actual_batch = execute_query_fetch_all(USE_CASE_4_QUERY)
    
    ray.shutdown()
    print("Query time", timer.total_elapsed_time)
    print(len(actual_batch))


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_4_balance_eddies(benchmark, setup_pytorch_tests):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 2)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Cost")
    config.update_value("experimental", "laminar_routing_policy", "AnyProxyBalance")

    context = Context()
    ray.init(num_gpus=len(context.gpus))

    # Actual query execution.
    config.update_value("experimental", "eddy", True)
    timer = Timer()
    with timer:
        actual_batch = execute_query_fetch_all(USE_CASE_4_QUERY)
        
    ray.shutdown()
    print("Query time", timer.total_elapsed_time)
    print(len(actual_batch))
