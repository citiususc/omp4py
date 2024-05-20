import os
import sys
import threading


class DummyCtx:

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class OpenMpLevel:
    def __init__(self, level: int, active_level: int, num_threads: int, thread_num: int,
                 barrier: threading.Barrier | None, lock: threading.RLock):
        self.level: int = level
        self.active_level: int = active_level
        self.num_threads: int = num_threads
        self.thread_num: int = thread_num
        self.barrier: threading.Barrier | None = barrier
        self.lock: threading.RLock = lock


class OpenMPContext:

    def __init__(self):
        self.local: threading.local = threading.local()
        self.lock: threading.RLock = threading.RLock()

        self.max_num_threads: int = self.num_procs()
        self.nested: bool = False
        self.schedule: str = "static"
        self.chunk_size: int = 0
        self.max_active_levels: int = sys.maxsize
        self.thread_limit: int = sys.maxsize  # TODO

        schedule = os.getenv("OMP_SCHEDULE")
        if schedule is not None:
            if "," in schedule:
                schedule, chunk_size = schedule.split(",")
                try:
                    chunk_size = int(chunk_size)
                except ValueError:
                    chunk_size = self.chunk_size
            else:
                chunk_size = self.chunk_size
            if schedule not in ["static", "dynamic", "guided", "auto"]:
                schedule = self.schedule
            self.schedule = schedule
            self.chunk_size = chunk_size

        max_num_threads = os.getenv("OMP_NUM_THREADS")
        if max_num_threads is not None:
            try:
                max_num_threads = int(max_num_threads)
                if max_num_threads > 0:
                    self.max_num_threads = max_num_threads
            except ValueError:
                pass

        dynamic = os.getenv("OMP_DYNAMIC")  # TODO

        nested = os.getenv("OMP_NESTED")
        if nested is not None:
            self.nested = nested.lower() == "true"

        stack_size = os.getenv("OMP_STACKSIZE")  # ignored
        wait_policy = os.getenv("OMP_WAIT_POLICY")  # ignored

        max_active_levels = os.getenv("OMP_MAX_ACTIVE_LEVELS")
        if max_active_levels is not None:
            try:
                max_active_levels = int(max_active_levels)
                if max_active_levels > 0:
                    self.max_active_levels = max_active_levels
            except ValueError:
                pass

        thread_limit = os.getenv("OMP_THREAD_LIMIT")
        if thread_limit is not None:
            try:
                thread_limit = int(thread_limit)
                if thread_limit > 0:
                    self.thread_limit = thread_limit
            except ValueError:
                pass

        self.set_levels([self.level0()])

    def level0(self) -> OpenMpLevel:
        return OpenMpLevel(
            level=0,
            active_level=0,
            num_threads=1,
            thread_num=0,
            barrier=None,
            lock=self.lock
        )

    def set_levels(self, levels: list[OpenMpLevel]):
        self.local.levels = levels

    def levels(self) -> list[OpenMpLevel]:
        try:
            return self.local.levels
        except:  # Omp call from a no omp or main thread
            self.local.levels = [self.level0()]
            return self.local.levels

    def current_level(self) -> OpenMpLevel:
        return self.levels()[-1]

    def get_level(self, level: int):
        return self.levels()[level]

    def num_procs(self) -> int:
        try:
            return len(os.sched_getaffinity(0))
        except:
            return os.cpu_count()
