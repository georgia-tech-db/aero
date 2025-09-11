git checkout artifacts-aubhro-2
conda create -n myenv python=3.9
conda activate myenv
pip install -e ".[dev]"
pytest -v test/benchmark_tests/eddies
