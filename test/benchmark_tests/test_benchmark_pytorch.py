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
from collections import Counter
from test.util import explain_query_plan

import pytest
import ray

from eva.experimental.eddy.util.profiler import Profiler
from eva.configuration.configuration_manager import ConfigurationManager
from eva.executor.execution_context import Context
from eva.server.command_handler import execute_query_fetch_all
from eva.utils.logging_manager import logger
from eva.utils.stats import Timer


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=1,
    min_rounds=5,
)
@pytest.mark.notparallel
def test_should_run_pytorch_and_yolo(benchmark, setup_pytorch_tests):
    select_query = """SELECT YoloV5(data) FROM MyVideo WHERE id = 1;"""
    # explain_select_query = """EXPLAIN {}""".format(select_query)
    # explain_batch = execute_query_fetch_all(explain_select_query)
    # print(explain_batch.frames.iloc[0][0])
    actual_batch = execute_query_fetch_all(select_query)
    print(actual_batch.frames)
    print(len(actual_batch.frames["yolov5.labels"][0]))
    print(actual_batch.frames["yolov5.labels"][0])
    print(len(actual_batch.frames["yolov5.bboxes"][0]))
    print(actual_batch.frames["yolov5.bboxes"][0])
    # assert len(actual_batch) == 5


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_should_run_pytorch_and_facenet(benchmark, setup_pytorch_tests):
    create_udf_query = """CREATE UDF IF NOT EXISTS FaceDetector
                INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),
                        scores NDARRAY FLOAT32(ANYDIM))
                TYPE  FaceDetection
                IMPL  'eva/udfs/face_detector.py';
    """
    execute_query_fetch_all(create_udf_query)

    select_query = """SELECT FaceDetector(data) FROM MyVideo
                    WHERE id < 5;"""

    actual_batch = benchmark(execute_query_fetch_all, select_query)
    assert len(actual_batch) == 5


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_should_run_pytorch_and_resnet50(benchmark, setup_pytorch_tests):
    create_udf_query = """CREATE UDF IF NOT EXISTS FeatureExtractor
                INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
                OUTPUT (features NDARRAY FLOAT32(ANYDIM))
                TYPE  Classification
                IMPL  'eva/udfs/feature_extractor.py';
    """
    execute_query_fetch_all(create_udf_query)

    select_query = """SELECT FeatureExtractor(data) FROM MyVideo
                    WHERE id < 5;"""
    actual_batch = benchmark(execute_query_fetch_all, select_query)
    assert len(actual_batch) == 5

    # non-trivial test case for Resnet50
    res = actual_batch.frames
    assert res["featureextractor.features"][0].shape == (1, 2048)
    # assert res["featureextractor.features"][0][0][0] > 0.3


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
@pytest.mark.notparallel
def test_lateral_join(benchmark, setup_pytorch_tests):
    select_query = """SELECT id, a FROM MyVideo JOIN LATERAL
                    YoloV5(data) AS T(a,b,c) WHERE id < 5;"""
    actual_batch = benchmark(execute_query_fetch_all, select_query)
    assert len(actual_batch) == 5
    assert list(actual_batch.columns) == ["myvideo.id", "T.a"]


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=1,
    min_rounds=1,
)
def test_dog_breed(benchmark, setup_pytorch_tests):
    # Document video statistics.
    # Video = BigSmallDogPlay
    #   Top dog breeds = great dane, chihuahua, german shepherd, labrador retriever, american staffordshire terrier
    #   Top dog colors = else, orange, black, gray, red
    # Video = BigSmallDogPlayShort
    #   Top dog breeds = great dane, chihuahua, american staffordshire terrier, boston bull, italian greyhound
    #   Top dog colors = else, orange, black, gray, red
    #   DogBreedClassifier = 194.90 sec
    #   Color = 163.27 sec

    ##########################
    # Update running mode
    ##########################
    ConfigurationManager().update_value("core", "mode", "release")

    ##########################
    # Set GPU configurations
    ##########################
    context = Context()
    if ConfigurationManager().get_value("experimental", "eddy"):
        ray.init(num_gpus=len(context.gpus))

    ##########################
    # Get basic stats
    ##########################
    # ConfigurationManager().update_value("experimental", "eddy", False)
    # ConfigurationManager().update_value("experimental", "eddy", True)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
    #                       WHERE Object.label = 'dog';"""
    # total = len(execute_query_fetch_all(select_query))
    # print("Total", total)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
    #                       WHERE Object.label = 'dog' AND Color(Crop(data, bbox)) = 'else';"""
    # print("Other", len(execute_query_fetch_all(select_query)) / total)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
    #                       WHERE Object.label = 'dog' AND Color(Crop(data, bbox)) = 'black';"""
    # print("Black", len(execute_query_fetch_all(select_query)) / total)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
    #                       WHERE Object.label = 'dog' AND Color(Crop(data, bbox)) = 'gray';"""
    # print("Red", len(execute_query_fetch_all(select_query)) / total)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
    #                       WHERE Object.label = 'dog' AND DogBreedClassifier(Crop(data, bbox)) = 'great dane';"""
    # print("Great dane", len(execute_query_fetch_all(select_query)) / total)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
    #                       WHERE Object.label = 'dog' AND DogBreedClassifier(Crop(data, bbox)) = 'labrador retriever';"""
    # print("Chihuahua", len(execute_query_fetch_all(select_query)) / total)

    ##########################
    # Warm up the cache
    ##########################
    # ConfigurationManager().update_value("experimental", "eddy", False)
    # select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
    #                       JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score);"""
    # execute_query_fetch_all(select_query)

    ##########################
    # Running eddy execution
    ##########################
    ConfigurationManager().update_value("experimental", "eddy", True)
    select_query = """SELECT id, bbox FROM BigSmallDogPlayVideo
                          JOIN LATERAL UNNEST(YoloV5(data)) AS Object(label, bbox, score) 
                          WHERE Object.label = 'dog'
                              AND DogBreedClassifier(Crop(data, bbox)) = 'great dane'
                              AND Color(Crop(data, bbox)) = 'black';"""

    timer = Timer()
    with timer:
        actual_batch = execute_query_fetch_all(select_query)
    print("Query time", timer.total_elapsed_time)

    print(len(actual_batch))


@pytest.mark.torchtest
@pytest.mark.benchmark(
    warmup=True,
    warmup_iterations=1,
    min_rounds=1,
)
def test_unsafe(benchmark, setup_pytorch_tests):
    context = Context()
    if ConfigurationManager().get_value("experimental", "eddy"):
        ray.init(num_gpus=len(context.gpus))

    timer = Timer()

    ##########################
    # Update running mode
    ##########################
    ConfigurationManager().update_value("core", "mode", "release")

    #######################
    # Warm up
    #######################
    # ConfigurationManager().update_value("experimental", "eddy", False)
    # select_query = f"""SELECT id, YoloV5(data).labels FROM WarehouseVideo WHERE id > 1000 AND id < 7000;"""
    # execute_query_fetch_all(select_query)
    # select_query = f"""SELECT id, HardHatDetector(data).labels FROM WarehouseVideo WHERE id > 8000 AND id < 14000;"""
    # execute_query_fetch_all(select_query)

    #######################
    # Profiler wrapper 
    #######################
    # num_gpus = ConfigurationManager().get_value("experimental", "logical_filter_to_physical_rule_gpus")
    # prof_list = [Profiler(i) for i in range(num_gpus)]
    # for prof in prof_list:
    #     prof.start()

    #######################
    # Actual query
    #######################
    ConfigurationManager().update_value("experimental", "eddy", True)
    select_query = """SELECT id FROM WarehouseVideo
                          WHERE ['person'] <@ YoloV5(data).labels
                              AND ['no hardhat'] <@ HardHatDetector(data).labels;"""

    with timer:
        actual_batch = execute_query_fetch_all(select_query)
    print("Query time", timer.total_elapsed_time)

    print(len(actual_batch))

    #######################
    # Print GPU util 
    #######################
    # for prof in prof_list:
    #     prof.stop()
    # gpu_util_list = [prof._gpu_util_rate for prof in prof_list]
    # for i, gpu_util in enumerate(gpu_util_list):
    #     with open(f"stats_gpu_util_{i}.txt", "w") as f:
    #         gpu_util = [f"{data_tuple[0]},{data_tuple[1]}" for data_tuple in gpu_util]
    #         f.write("\n".join(gpu_util))
    #         f.flush()


@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
def test_automatic_speech_recognition(benchmark, setup_pytorch_tests):
    udf_name = "SpeechRecognizer"
    create_udf = (
        f"CREATE UDF {udf_name} TYPE HuggingFace "
        "'task' 'automatic-speech-recognition' 'model' 'openai/whisper-base';"
    )
    execute_query_fetch_all(create_udf)

    # TODO: use with SAMPLE AUDIORATE 16000
    select_query = f"SELECT {udf_name}(audio) FROM VIDEOS;"
    output = benchmark(execute_query_fetch_all(select_query))

    # verify that output has one row and one column only
    assert output.frames.shape == (1, 1)
    # verify that speech was converted to text correctly
    assert output.frames.iloc[0][0].count("touchdown") == 2

    drop_udf_query = f"DROP UDF {udf_name};"
    execute_query_fetch_all(drop_udf_query)


@pytest.mark.benchmark(
    warmup=False,
    warmup_iterations=1,
    min_rounds=1,
)
def test_summarization_from_video(benchmark, setup_pytorch_tests):
    asr_udf = "SpeechRecognizer"
    create_udf = (
        f"CREATE UDF {asr_udf} TYPE HuggingFace "
        "'task' 'automatic-speech-recognition' 'model' 'openai/whisper-base';"
    )
    execute_query_fetch_all(create_udf)

    summary_udf = "Summarizer"
    create_udf = (
        f"CREATE UDF {summary_udf} TYPE HuggingFace "
        "'task' 'summarization' 'model' 'philschmid/bart-large-cnn-samsum' 'min_length' 10 'max_length' 100;"
    )
    execute_query_fetch_all(create_udf)

    # TODO: use with SAMPLE AUDIORATE 16000
    select_query = f"SELECT {summary_udf}({asr_udf}(audio)) FROM VIDEOS;"
    output = benchmark(execute_query_fetch_all(select_query))

    # verify that output has one row and one column only
    assert output.frames.shape == (1, 1)
    # verify that summary is as expected
    assert (
        output.frames.iloc[0][0]
        == "Jalen Hurts has scored his second rushing touchdown of the game."
    )

    drop_udf_query = f"DROP UDF {asr_udf};"
    execute_query_fetch_all(drop_udf_query)
    drop_udf_query = f"DROP UDF {summary_udf};"
    execute_query_fetch_all(drop_udf_query)
