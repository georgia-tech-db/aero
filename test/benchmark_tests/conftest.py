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
from test.util import load_udfs_for_testing

import pytest
import ray

from eva.catalog.catalog_manager import CatalogManager
from eva.configuration.constants import EVA_ROOT_DIR
from eva.executor.execution_context import Context
from eva.server.command_handler import execute_query_fetch_all
from eva.udfs.udf_bootstrap_queries import init_builtin_video_udfs, init_llm_udf

@pytest.fixture(autouse=True)
def reset_catalogue():
    CatalogManager().reset()


@pytest.fixture(autouse=False)
def load_dog_videos():
    dog_video_path = f"{EVA_ROOT_DIR}/data/big-small-dog-play/big-small-dog-play-super-short.mp4"
    load_video_query = f"LOAD VIDEO '{dog_video_path}' INTO ShortBigSmallDogPlayVideo;"
    execute_query_fetch_all(load_video_query)

    dog_video_path = f"{EVA_ROOT_DIR}/data/big-small-dog-play/big-small-dog-play-short.mp4"
    load_video_query = f"LOAD VIDEO '{dog_video_path}' INTO BigSmallDogPlayVideo;"
    execute_query_fetch_all(load_video_query)

    dog_video_path = f"{EVA_ROOT_DIR}/data/big-small-dog-play/big-small-dog-play.mp4"
    load_video_query = f"LOAD VIDEO '{dog_video_path}' INTO BigSmallDogPlayVideoLong;"
    execute_query_fetch_all(load_video_query)


@pytest.fixture(autouse=False)
def load_warehouse_videos():
    warehouse_video_path = f"{EVA_ROOT_DIR}/data/safety/factory.mp4"
    load_video_query = f"LOAD VIDEO '{warehouse_video_path}' INTO WarehouseVideo;"
    execute_query_fetch_all(load_video_query)

    warehouse_video_path = f"{EVA_ROOT_DIR}/data/safety/factory-long-duplicate.mp4"
    load_video_query = f"LOAD VIDEO '{warehouse_video_path}' INTO LongWarehouseVideo;"
    execute_query_fetch_all(load_video_query)

@pytest.fixture(autouse=False)
def load_foodreview_text():
    execute_query_fetch_all("""CREATE TABLE IF NOT EXISTS FoodReview (rating INTEGER, review TEXT(10000));""")
    food_review_path = f"{EVA_ROOT_DIR}/data/food_reviews/normal.txt"
    load_csv_query = f"LOAD CSV '{food_review_path}' INTO FoodReview;" 
    execute_query_fetch_all(load_csv_query)

    yield

    execute_query_fetch_all("""DROP TABLE IF EXISTS FoodReview;""")


@pytest.fixture(autouse=False)
def load_video_udfs():
    init_builtin_video_udfs()


@pytest.fixture(autouse=False)
def load_llm_udf():
    init_llm_udf()


@pytest.fixture(autouse=False)
def ray_fixture():
    context = Context()
    ray.init(num_gpus=len(context.gpus))
    yield
    ray.shutdown()
