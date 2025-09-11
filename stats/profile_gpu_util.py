import os
import argparse
import subprocess as subp

from eva.experimental.eddy.util.profiler import Profiler


def run_experiment(num_gpus):
    prof_list = [Profiler(i) for i in range(num_gpus)]
    for prof in prof_list:
        prof.start()
    subp.Popen(
        "pytest test/benchmark_tests/test_benchmark_pytorch.py -s -k test_unsafe".split(" "),

    ).wait()
    for prof in prof_list:
        prof.stop()
    return [
        prof._gpu_util_rate for prof in prof_list
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-gpus", type=int, default=1)
    args = parser.parse_args()

    # Run experiment.
    gpu_util_list = run_experiment(args.num_gpus)

    # Write GPU utilization.
    for i, gpu_util in enumerate(gpu_util_list):
        with open(f"stats_gpu_util_{i}.txt", "w") as f:
            gpu_util = [f"{data_tuple[0]},{data_tuple[1]}" for data_tuple in gpu_util]
            f.write("\n".join(gpu_util))
            f.flush()


if __name__ == "__main__":
    main()
