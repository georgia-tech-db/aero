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


QUERY = "uc1_3"
ENABLE_CACHE = False


def get_query_and_predicate_based_on_config():
    config = ConfigurationManager()
    query = config.get_value("experimental", "query")
    if query == "uc1_1":
        return (
            """SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                    WHERE Object.label = 'dog'""",
            "DogBreedClassifier(Crop(data, bbox)) = 'great dane'",
            "Color(Crop(data, bbox)) = 'black'",
        )
    elif query == "uc1_2":
        return (
            """SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                    WHERE Object.label = 'dog'""",
            "DogBreedClassifier(Crop(data, bbox)) = 'labrador retriever'",
            "Color(Crop(data, bbox)) = 'other'",
        )
    elif query == "uc1_3":
        return (
            """SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                    WHERE Object.label = 'dog'""",
            "DogBreedClassifier(Crop(data, bbox)) = 'great dane'",
            "Color(Crop(data, bbox)) = 'gray'",
        )
    else:
        raise Exception(f"Query {query} is not implemented for UC1")


def get_query_based_on_config():
    config = ConfigurationManager()
    query = config.get_value("experimental", "query")
    if query == "uc1_1":
        return """SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                    WHERE Object.label = 'dog'
                      AND DogBreedClassifier(Crop(data, bbox)) = 'great dane'
                      AND Color(Crop(data, bbox)) = 'black';"""
    elif query == "uc1_2":
        return """SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                    WHERE Object.label = 'dog'
                      AND DogBreedClassifier(Crop(data, bbox)) = 'labrador retriever'
                      AND Color(Crop(data, bbox)) = 'other';"""
    elif query == "uc1_3":
        return """SELECT id, bbox FROM BigSmallDogPlayVideo
                    JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                    WHERE Object.label = 'dog'
                      AND DogBreedClassifier(Crop(data, bbox)) = 'great dane'
                      AND Color(Crop(data, bbox)) = 'gray';"""
    elif query == "uc1_4":
        config.update_value("experimental", "synthetic_udf1_cost", 0.01)
        config.update_value("experimental", "synthetic_udf1_sel", 0.4)
        config.update_value("experimental", "synthetic_udf2_cost", 0.02)
        config.update_value("experimental", "synthetic_udf2_sel", 0.9)
        EVA_INSTALLATION_DIR = config.get_value("core", "eva_installation_dir")
        execute_query_fetch_all(
            """CREATE UDF IF NOT EXISTS SyntheticUDF1
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (labels NDARRAY STR(ANYDIM))
                  TYPE  Classification
                  IMPL  '{}/udfs/synthetic_udf1.py';
        """.format(
                EVA_INSTALLATION_DIR
            )
        )
        execute_query_fetch_all(
            """CREATE UDF IF NOT EXISTS SyntheticUDF2
                  INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                  OUTPUT (labels NDARRAY STR(ANYDIM))
                  TYPE  Classification
                  IMPL  '{}/udfs/synthetic_udf2.py';
        """.format(
                EVA_INSTALLATION_DIR
            )
        )
        return """SELECT * FROM BigSmallDogPlayVideo
                    WHERE ['True'] <@ SyntheticUDF1(data).labels
                      AND ['True'] <@ SyntheticUDF2(data).labels"""
    else:
        raise Exception(f"Query {query} is not implemented for UC1")


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_1_baseline(benchmark, load_dog_videos, load_video_udfs):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "query", QUERY)
    config.update_value("experimental", "eddy", False)
    config.update_value("experimental", "cache", ENABLE_CACHE)

    if config.get_value("experimental", "cache"):
        execute_query_fetch_all(get_query_and_predicate_based_on_config()[0])
    
    benchmark(execute_query_fetch_all, get_query_based_on_config())


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_1_best_reorder(benchmark, load_dog_videos, load_video_udfs):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "query", QUERY)
    config.update_value("experimental", "eddy", False)
    config.update_value("experimental", "cache", ENABLE_CACHE)

    if config.get_value("experimental", "cache"):
        execute_query_fetch_all(get_query_and_predicate_based_on_config()[0])

    (
        base_query,
        dog_breed_pred,
        dog_color_pred,
    ) = get_query_and_predicate_based_on_config()
    query = base_query + " AND {} AND {}".format(dog_color_pred, dog_breed_pred)
    
    benchmark(execute_query_fetch_all, query)


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_1_eddies_cost_driven(benchmark, load_dog_videos, load_video_udfs, ray_fixture):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "query", QUERY)
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "cache", ENABLE_CACHE)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 1)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Cost")
    config.update_value("experimental", "laminar_routing_policy", "AnyRoundRobin")

    if config.get_value("experimental", "cache"):
        execute_query_fetch_all(get_query_and_predicate_based_on_config()[0])

    benchmark(execute_query_fetch_all, get_query_based_on_config())


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_1_eddies_selectivity_driven(benchmark, load_dog_videos, load_video_udfs, ray_fixture):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "query", QUERY)
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "cache", ENABLE_CACHE)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 1)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Selectivity")
    config.update_value("experimental", "laminar_routing_policy", "AnyRoundRobin")

    if config.get_value("experimental", "cache"):
        execute_query_fetch_all(get_query_and_predicate_based_on_config()[0])

    benchmark(execute_query_fetch_all, get_query_based_on_config())


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_1_eddies_score_driven(benchmark, load_dog_videos, load_video_udfs, ray_fixture):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "query", QUERY)
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "cache", ENABLE_CACHE)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 1)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 1)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)
    config.update_value("experimental", "eddy_routing_policy", "AllCost")
    config.update_value("experimental", "eddy_ranking_function", "Score")
    config.update_value("experimental", "laminar_routing_policy", "AnyRoundRobin")

    if config.get_value("experimental", "cache"):
        execute_query_fetch_all(get_query_and_predicate_based_on_config()[0])

    benchmark(execute_query_fetch_all, get_query_based_on_config())


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_use_case_1_eddies_multi_gpu(benchmark, load_dog_videos, load_video_udfs, ray_fixture):
    config = ConfigurationManager()
    config.update_value("core", "mode", "debug")
    config.update_value("experimental", "query", QUERY)
    config.update_value("experimental", "eddy", True)
    config.update_value("experimental", "cache", ENABLE_CACHE)
    config.update_value("experimental", "logical_filter_to_physical_rule_workers", 2)
    config.update_value("experimental", "logical_filter_to_physical_rule_gpus", 2)
    config.update_value("experimental", "logical_get_to_sequential_scan_workers", 2)
    config.update_value("experimental", "logical_get_to_sequential_scan_gpus", 1)

    if config.get_value("experimental", "cache"):
        execute_query_fetch_all(get_query_and_predicate_based_on_config()[0])
    
    benchmark(execute_query_fetch_all, get_query_based_on_config())
