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


@pytest.fixture(autouse=False)
def setup_pytorch_tests():
    CatalogManager().reset()
    # execute_query_fetch_all("LOAD VIDEO 'data/ua_detrac/ua_detrac.mp4' INTO MyVideo;")
    # execute_query_fetch_all("LOAD VIDEO 'data/mnist/mnist.mp4' INTO MNIST;")

    dog_video_path = f"{EVA_ROOT_DIR}/data/big-small-dog-play/big-small-dog-play-super-short.mp4"
    load_video_query = f"LOAD VIDEO '{dog_video_path}' INTO ShortBigSmallDogPlayVideo;"
    execute_query_fetch_all(load_video_query)

    dog_video_path = f"{EVA_ROOT_DIR}/data/big-small-dog-play/big-small-dog-play-short.mp4"
    load_video_query = f"LOAD VIDEO '{dog_video_path}' INTO BigSmallDogPlayVideo;"
    execute_query_fetch_all(load_video_query)

    dog_video_path = f"{EVA_ROOT_DIR}/data/big-small-dog-play/big-small-dog-play.mp4"
    load_video_query = f"LOAD VIDEO '{dog_video_path}' INTO BigSmallDogPlayVideoLong;"
    execute_query_fetch_all(load_video_query)

    warehouse_video_path = f"{EVA_ROOT_DIR}/data/safety/factory.mp4"
    load_video_query = f"LOAD VIDEO '{warehouse_video_path}' INTO WarehouseVideo;"
    execute_query_fetch_all(load_video_query)

    warehouse_video_path = f"{EVA_ROOT_DIR}/data/safety/factory-long-duplicate.mp4"
    load_video_query = f"LOAD VIDEO '{warehouse_video_path}' INTO LongWarehouseVideo;"
    execute_query_fetch_all(load_video_query)

    execute_query_fetch_all("""CREATE TABLE IF NOT EXISTS FoodReview (rating INTEGER, review TEXT(10000));""")
    food_review_path = f"{EVA_ROOT_DIR}/data/food_reviews/normal.txt"
    load_csv_query = f"LOAD CSV '{food_review_path}' INTO FoodReview;" 
    execute_query_fetch_all(load_csv_query)

    load_udfs_for_testing()
    yield None
    
    execute_query_fetch_all("""DROP TABLE IF EXISTS FoodReview;""")

def ray_fixture():
    context = Context()
    ray.init(num_gpus=len(context.gpus))
    yield
    ray.shutdown()