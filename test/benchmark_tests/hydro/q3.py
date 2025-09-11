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

from stats.profile_gpu_util import Profiler


QUERY = """SELECT * FROM WarehouseVideo
    WHERE ['person'] <@ YoloV5(data).labels
    AND ['no hardhat'] <@ HardHatDetector(data).labels;"""


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_evadb(benchmark, setup_pytorch_tests):
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
        actual_batch = execute_query_fetch_all(QUERY)
    print("Query time", timer.total_elapsed_time)
    print(len(actual_batch))


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_hydro(benchmark, setup_pytorch_tests):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 50)
    config.update_value(
        "experimental", "logical_filter_to_physical_rule_gpus", 1 
    )
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Cost")
    config.update_value(
        "experimental", "laminar_routing_policy", "AnyDynamicRoundRobin"
    )

    context = Context()
    ray.init(num_gpus=len(context.gpus))

    timer = Timer()

    # Actual query execution.
    config.update_value("experimental", "eddy", True)
    with timer:
        actual_batch = execute_query_fetch_all(QUERY)
    print("Query time", timer.total_elapsed_time)
    print(len(actual_batch))


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_hydro_scale(benchmark, setup_pytorch_tests):
    NUM_GPUS = 1

    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 50)
    config.update_value(
        "experimental", "logical_filter_to_physical_rule_gpus", NUM_GPUS
    )
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Cost")
    config.update_value(
        "experimental", "laminar_routing_policy", "AnyDynamicRoundRobin"
    )

    ray.init(num_gpus=NUM_GPUS)

    timer = Timer()

    # Actual query execution.
    config.update_value("experimental", "eddy", True)
    with timer:
        prof_list = [Profiler(i) for i in range(NUM_GPUS)]
        for prof in prof_list:
            prof.start()

        # Query execution
        actual_batch = execute_query_fetch_all(QUERY)

        for prof in prof_list:
            prof.stop()
        gpu_util_list = [prof._gpu_util_rate for prof in prof_list]
        for i, gpu_util in enumerate(gpu_util_list):
            with open(f"stats_gpu_util_{i}.txt", "w") as f:
                gpu_util = [f"{data_tuple[0]},{data_tuple[1]}" for data_tuple in gpu_util]
                f.write("\n".join(gpu_util))
                f.flush()
    print("Query time", timer.total_elapsed_time)
    print(len(actual_batch))
