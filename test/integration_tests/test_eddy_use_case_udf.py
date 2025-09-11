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
import unittest
from test.util import load_udfs_for_testing

from eva.catalog.catalog_manager import CatalogManager
from eva.configuration.configuration_manager import ConfigurationManager
from eva.configuration.constants import EVA_ROOT_DIR
from eva.server.command_handler import execute_query_fetch_all


class EddyUseCaseUdfTests(unittest.TestCase):
    def setUp(self):
        CatalogManager().reset()
        ConfigurationManager()
        # Load built-in UDFs.
        load_udfs_for_testing()

        # Load color images.
        load_query = (
            f"LOAD IMAGE '{EVA_ROOT_DIR}/data/color/pink.jpg' INTO testColorTable;"
        )
        execute_query_fetch_all(load_query)
        load_query = (
            f"LOAD IMAGE '{EVA_ROOT_DIR}/data/color/green.jpg' INTO testColorTable;"
        )
        execute_query_fetch_all(load_query)
        load_query = (
            f"LOAD IMAGE '{EVA_ROOT_DIR}/data/color/orange.jpg' INTO testColorTable;"
        )
        execute_query_fetch_all(load_query)

        load_query = f"LOAD VIDEO '{EVA_ROOT_DIR}/data/color/color.mp4' INTO testColorVideoTable;"
        execute_query_fetch_all(load_query)

        # Load hard hat images.
        load_query = f"LOAD VIDEO '{EVA_ROOT_DIR}/data/hardhat/hardhat.mp4' INTO testHardHatTable;"
        execute_query_fetch_all(load_query)

        # Load scene images.
        load_query = (
            f"LOAD VIDEO '{EVA_ROOT_DIR}/data/scene/warehouse.mp4' INTO testSceneTable;"
        )
        execute_query_fetch_all(load_query)

    def tearDown(self):
        # Drop table.
        drop_table_query = "DROP TABLE testColorTable;"
        execute_query_fetch_all(drop_table_query)
        drop_table_query = "DROP TABLE testColorVideoTable;"
        execute_query_fetch_all(drop_table_query)
        drop_table_query = "DROP TABLE testHardHatTable;"
        execute_query_fetch_all(drop_table_query)

    def test_correct_color_from_image(self):
        select_query = "SELECT name, Color(data) FROM testColorTable;"
        actual_batch = execute_query_fetch_all(select_query)

        df = actual_batch.frames
        for i in range(len(actual_batch)):
            color = df["color.color"][i].lower()
            path = df["testcolortable.name"][i].lower()
            self.assertIn(color, path)

    def test_correct_color_from_video(self):
        select_query = "SELECT id, Color(data) FROM testColorVideoTable;"
        actual_batch = execute_query_fetch_all(select_query)

        df = actual_batch.frames
        for i in range(len(actual_batch)):
            color = df["color.color"][i].lower()
            print(color)

    def test_hard_hat_from_image(self):
        select_query = "SELECT HardHatDetector(data) FROM testHardHatTable;"
        actual_batch = execute_query_fetch_all(select_query)

        df = actual_batch.frames
        print(df)
        # for i in range(len(actual_batch)):
        #     color = df["color.color"][i].lower()
        #     print(color)

    def test_scene_from_image(self):
        select_query = "SELECT SceneClassifier(data) FROM testSceneTable;"
        actual_batch = execute_query_fetch_all(select_query)

        df = actual_batch.frames
        print(df)
        # for i in range(len(actual_batch)):
        #     color = df["color.color"][i].lower()
        #     print(color)
