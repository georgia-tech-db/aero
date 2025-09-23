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

from eva.configuration.configuration_manager import ConfigurationManager
from eva.server.command_handler import execute_query_fetch_all
from eva.utils.stats import Timer


NUM_GPUS = 1
USE_CASE_5_QUERY = f"""SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score)
                    WHERE Object.label = 'dog'
                      AND DogBreedClassifier(Crop(data, bbox)) = 'great dane'
                      AND Area(data, bbox) > 0.3
                      AND Color(Crop(data, bbox)) = 'black';"""


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_5_baseline(benchmark, load_dog_videos, load_video_udfs):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "eddy", False)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 1)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)

    execute_query_fetch_all("""SELECT YoloV5(data) FROM BigSmallDogPlayVideo""")

    benchmark(execute_query_fetch_all, USE_CASE_5_QUERY)


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_5_hydro(benchmark, load_dog_videos, load_video_udfs, ray_fixture):
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

    execute_query_fetch_all("""SELECT YoloV5(data) FROM BigSmallDogPlayVideo""")

    benchmark(execute_query_fetch_all, USE_CASE_5_QUERY)
