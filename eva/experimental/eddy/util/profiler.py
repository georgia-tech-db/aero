import pynvml
import threading

from time import perf_counter


class Profiler:
    def __init__(self, gpu_idx=0):
        pynvml.nvmlInit()
        assert gpu_idx >= 0
        self._hdl = pynvml.nvmlDeviceGetHandleByIndex(gpu_idx)

        # GPU utilization and memory usage.
        self._gpu_util_rate = []

        self._should_stop_cond = threading.Condition()

    def start(self):
        threading.Thread(target=self._profile).start()

    def stop(self):
        with self._should_stop_cond:
            self._should_stop_cond.notify()

    @property
    def actual_gpu_util(self):
        gpu_util = []
        for util in self._gpu_util_rate:
            if util > 0:
                gpu_util.append(util)
        return -1 if len(gpu_util) == 0 else sum(gpu_util) / len(gpu_util) / 100

    @property
    def temporal_gpu_util(self):
        gpu_util = []
        for util in self._gpu_util_rate:
            gpu_util.append(1 if util > 0 else 0)
        return -1 if len(gpu_util) == 0 else sum(gpu_util) / len(gpu_util)

    @property
    def memory_usage(self):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._hdl)
        return mem.used / mem.total


    def reset(self):
        self._gpu_util_rate = []

    def _profile(self):
        st = perf_counter()
        with self._should_stop_cond:
            while True:
                if self._should_stop_cond.wait(timeout=0.001):
                    break
                usage = pynvml.nvmlDeviceGetUtilizationRates(self._hdl)
                self._gpu_util_rate.append((perf_counter() - st, usage.gpu))
