"""Basic atomic synchronization primitives used by the `omp4py` runtime.

This module provides simple atomic types implemented using Python locks.
They are used internally by the runtime to implement synchronization
patterns required by OpenMP constructs.

The provided primitives include:

- `AtomicFlag`: A boolean flag with an atomic `test_and_set` operation.
- `AtomicInt`: An integer value supporting common atomic operations such
  as exchange, compare-and-exchange, and arithmetic updates.
- `AtomicObject`: A container that allows a value to be set atomically
  once.

These implementations are designed to behave similarly to their C atomic
counterparts while remaining compatible with the pure Python runtime and
the compiled runtime.
"""
from __future__ import annotations

import threading
import typing

if typing.TYPE_CHECKING:
    from omp4py.runtime.lowlevel.numeric import pyint

__all__ = ["AtomicFlag", "AtomicInt", "AtomicObject"]


class AtomicFlag:
    """Atomic boolean flag used for lightweight synchronization."""

    _value: bool
    _lock: threading.Lock

    @staticmethod
    def new() -> AtomicFlag:
        """Create a new `AtomicFlag`.

        Returns:
            AtomicFlag: A new flag initialized to `False`.
        """
        return AtomicFlag()

    def __init__(self) -> None:
        """Initialize the flag."""
        self._value = False
        self._lock = threading.Lock()

    def test_and_set(self) -> bool:
        """Atomically set the flag and return the previous state.

        Returns:
            bool: `True` if the flag was already set, `False` otherwise.
        """
        if self._value:
            return True
        with self._lock:
            if not self._value:
                self._value = True
                return False
        return True

    def clear(self) -> None:
        """Reset the flag to `False`."""
        self._value = False


class AtomicInt:
    """Atomic integer supporting common synchronization operations."""

    _value: pyint
    _lock: threading.Lock

    @staticmethod
    def new(value: pyint = 0) -> AtomicInt:
        """Create a new `AtomicInt`.

        Args:
            value (pyint): Initial value.

        Returns:
            AtomicInt: New atomic integer.
        """
        return AtomicInt(value)

    def __init__(self, value: pyint = 0) -> None:
        """Initialize the atomic integer.

        Args:
            value (pyint): Initial value.
        """
        self._value = value
        self._lock = threading.Lock()

    def set(self, value: pyint) -> None:
        """Set the stored value.

        Args:
            value (pyint): New value.
        """
        self._value = value

    def get(self) -> pyint:
        """Return the current value.

        Returns:
            pyint: Current integer value.
        """
        return self._value

    def exchange(self, desired: pyint) -> pyint:
        """Atomically replace the value.

        Args:
            desired (pyint): New value.

        Returns:
            pyint: Previous value.
        """
        with self._lock:
            self._value, desired = desired, self._value
            return self._value

    def compare_exchange_strong(self, expected: pyint, desired: pyint) -> bool:
        """Replace the value if it equals `expected`.

        Args:
            expected (pyint): Expected current value.
            desired (pyint): New value if comparison succeeds.

        Returns:
            bool: `True` if the value was replaced.
        """
        with self._lock:
            if self._value == expected:
                self._value, desired = desired, self._value
                return True
            return False

    def compare_exchange_weak(self, expected: pyint, desired: pyint) -> bool:
        """Weak version of compare-and-exchange.

        Args:
            expected (pyint): Expected current value.
            desired (pyint): New value if comparison succeeds.

        Returns:
            bool: `True` if the value was replaced.
        """
        with self._lock:
            if self._value == expected:
                self._value, desired = desired, self._value
                return True
            return False

    def add(self, arg: pyint) -> pyint:
        """Atomically add to the value.

        Args:
            arg (pyint): Value to add.

        Returns:
            pyint: Updated value.
        """
        with self._lock:
            self._value += arg
            return self._value

    def sub(self, arg: pyint) -> pyint:
        """Atomically subtract from the value.

        Args:
            arg (pyint): Value to subtract.

        Returns:
            pyint: Updated value.
        """
        with self._lock:
            self._value -= arg
            return self._value

    def or_(self, arg: pyint) -> pyint:
        """Atomically apply bitwise OR.

        Args:
            arg (pyint): Operand.

        Returns:
            pyint: Updated value.
        """
        with self._lock:
            self._value = self._value | arg
            return self._value

    def xor(self, arg: pyint) -> pyint:
        """Atomically apply bitwise XOR.

        Args:
            arg (pyint): Operand.

        Returns:
            pyint: Updated value.
        """
        with self._lock:
            self._value = self._value ^ arg
            return self._value

    def and_(self, arg: pyint) -> pyint:
        """Atomically apply bitwise AND.

        Args:
            arg (pyint): Operand.

        Returns:
            pyint: Updated value.
        """
        with self._lock:
            self._value = self._value & arg
            return self._value


class AtomicObject[T]:
    """Atomic container that allows a value to be set atomically once."""

    _value: T | None
    _lock: threading.Lock

    @staticmethod
    def new() -> AtomicObject:
        """Create a new empty `AtomicObject`.

        Returns:
            AtomicObject: New container with no value set.
        """
        return AtomicObject()

    def __init__(self) -> None:
        """Initialize the container."""
        self._value = None
        self.lock = threading.Lock()

    def get(self) -> T | None:
        """Return the stored value.

        Returns:
            T | None: Stored value, or `None` if not set.
        """
        return self._value

    def set(self, value: T) -> bool:
        """Atomically set the value if it has not been assigned yet.

        Args:
            value (T): Value to store.

        Returns:
            bool: `True` if the value was stored, `False` otherwise.
        """
        if self._value is not None:
            return False
        with self.lock:
            if self._value is None:
                self._value = value
                return True
            return False
