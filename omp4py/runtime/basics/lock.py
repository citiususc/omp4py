import threading

from omp4py.runtime.basics.types import *


class Mutex:
    _lock: threading.Lock

    @staticmethod
    def new() -> 'Mutex':
        return Mutex()

    def __init__(self):
        self._lock = threading.Lock()

    def lock(self) -> None:
        self._lock.acquire()

    def unlock(self) -> None:
        self._lock.release()

    def test(self) -> bool:
        return self._lock.acquire(blocking=False)

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()
        return False


class RMutex:
    _lock: threading.RLock

    @staticmethod
    def new() -> 'RMutex':
        return RMutex()

    def __init__(self):
        self._lock = threading.RLock()

    def lock(self) -> None:
        self._lock.acquire()

    def unlock(self) -> None:
        self._lock.release()

    def test(self) -> bool:
        return self._lock.acquire(blocking=False)

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()
        return False


class Barrier:
    _value: threading.Barrier

    @staticmethod
    def new(parties: pyint) -> 'Barrier':
        return Barrier(parties)

    def __init__(self, parties: pyint):
        self._value = threading.Barrier(parties)

    def wait(self) -> None:
        self._value.wait()
