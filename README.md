# Aero: Adaptive Query Processing of ML Queries

We have created this archive for availability and partial reproducibility. Most experiments can be run automatically but use case 4 will require some effort to set up properly.

We have included the scripts to generate the plots from raw data files but as they were gathered manually, reproducing the data files from experimental outputs will require manual work.

To run the experiments for use case 1, 2, 3, 5 as laid out in the paper, do the following.

You must be on a machine running linux that has NVidia GPUs and has conda installed.

Before getting started, you must download the necessary videos and store them at `data/`. They can be found at https://drive.google.com/drive/folders/1Dpt2jzL3Libr2eKoEI71oSq06F-t3SUt

```sh

# newer versions of python have breaking changes
$ conda create -n myenv python=3.9

$ conda activate myenv
pip install -e ".[dev]"

# Use case 1
$ pytest test/benchmark_tests/eddies/test_use_case_1.py

# Use case 2
$ pytest test/benchmark_tests/eddies/test_use_case_2.py

# Use case 3
$ pytest test/benchmark_tests/eddies/test_use_case_3.py

# Use case 4
$ pytest test/benchmark_tests/eddies/test_use_case_4.py

# Use case 5
$ pytest test/benchmark_tests/eddies/test_use_case_5.py
```

## Benchmark
In order to save experiments, run the following command.
```ssh
$ pytest test/benchmark_tests/eddies --benchmark-save-data
```
