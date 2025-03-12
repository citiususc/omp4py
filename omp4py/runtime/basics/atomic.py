import typing
import threading
from omp4py.runtime.basics.types import *

__all__ = ['AtomicObject', 'AtomicInt', 'AtomicFlag']


class AtomicFlag:
    _value: bool
    _lock: threading.Lock

    @staticmethod
    def new() -> 'AtomicFlag':
        return AtomicFlag()

    def __init__(self):
        self._value = False

    def no_clear_test_and_set(self) -> bool:
        if self._value:
            return False
        return self.test_and_set()

    def test_and_set(self) -> bool:
        with self._lock:
            if not self._value:
                self._value = True
                return True
        return False

    def clear(self) -> None:
        self._value = False


class AtomicObject:
    _value: typing.Any
    _lock: threading.Lock

    @staticmethod
    def new() -> 'AtomicObject':
        return AtomicObject()

    def __init__(self):
        self._value = None
        self.lock = threading.Lock()

    def get(self) -> typing.Any:
        return self._value

    def set(self, value: typing.Any) -> bool:
        if self._value is not None:
            return False
        with self.lock:
            if self._value is None:
                self._value = value
                return True
            return False


class AtomicInt:
    _value: pyint
    _lock: threading.Lock

    @staticmethod
    def new(value: pyint = 0) -> 'AtomicInt':
        return AtomicInt(value)

    def __init__(self, value: pyint = 0):
        self._value = value
        self._lock = threading.Lock()

    def set(self, value: pyint):
        self._value = value

    def get(self) -> pyint:
        return self._value

    def exchange(self, desired: pyint) -> pyint:
        with self._lock:
            self._value, desired = desired, self._value
            return self._value

    def compare_exchange_strong(self, expected: pyint, desired: pyint) -> bool:
        with self._lock:
            if self._value == expected:
                self._value, desired = desired, self._value
                return True
            return False

    def compare_exchange_weak(self, expected: pyint, desired: pyint) -> bool:
        with self._lock:
            if self._value == expected:
                self._value, desired = desired, self._value
                return True
            return False

    def add(self, arg: pyint) -> pyint:
        with self._lock:
            self._value += arg
            return self._value

    def sub(self, arg: pyint) -> pyint:
        with self._lock:
            self._value -= arg
            return self._value

    def or_(self, arg: pyint) -> pyint:
        with self._lock:
            self._value = self._value | arg
            return self._value

    def xor(self, arg: pyint) -> pyint:
        with self._lock:
            self._value = self._value ^ arg
            return self._value

    def and_(self, arg: pyint) -> pyint:
        with self._lock:
            self._value = self._value & arg
            return self._value
