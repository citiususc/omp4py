"""Basic synchronization primitives used by the `omp4py` runtime.

This module provides simple wrappers around Python's `threading`
synchronization objects. These wrappers expose a small and consistent
API used by the runtime and mirror the primitives used in the compiled
runtime implementation.

The following primitives are defined:

- `Mutex`: Non-reentrant mutual exclusion lock.
- `RMutex`: Reentrant mutual exclusion lock.
- `Event`: Thread synchronization event used for signaling.

"""
from __future__ import annotations

import threading


class Mutex:
    """Non-reentrant mutual exclusion lock."""

    _lock: threading.Lock

    @staticmethod
    def new() -> Mutex:
        """Create a new `Mutex`.

        Returns:
            Mutex: New mutex instance.
        """
        return Mutex()

    def __init__(self) -> None:
        """Initialize the mutex."""
        self._lock = threading.Lock()

    def lock(self) -> None:
        """Acquire the mutex."""
        self._lock.acquire()

    def unlock(self) -> None:
        """Release the mutex."""
        self._lock.release()

    def test(self) -> bool:
        """Attempt to acquire the mutex without blocking.

        Returns:
            bool: `True` if the lock was acquired.
        """
        return self._lock.acquire(blocking=False)

    def __enter__(self) -> None:
        """Enter the mutex context."""
        self.lock()

    def __exit__(self, *_) -> bool:  # noqa: ANN002
        """Exit the mutex context."""
        self.unlock()
        return False


class RMutex:
    """Reentrant mutual exclusion lock."""

    _lock: threading.RLock

    @staticmethod
    def new() -> RMutex:
        """Create a new `RMutex`.

        Returns:
            RMutex: New reentrant mutex instance.
        """
        return RMutex()

    def __init__(self) -> None:
        """Initialize the reentrant mutex."""
        self._lock = threading.RLock()

    def lock(self) -> None:
        """Acquire the mutex."""
        self._lock.acquire()

    def unlock(self) -> None:
        """Release the mutex."""
        self._lock.release()

    def test(self) -> bool:
        """Attempt to acquire the mutex without blocking.

        Returns:
            bool: `True` if the lock was acquired.
        """
        return self._lock.acquire(blocking=False)

    def __enter__(self) -> None:
        """Enter the mutex context."""
        self.lock()

    def __exit__(self, *_) -> bool:  # noqa: ANN002
        """Exit the mutex context."""
        self.unlock()
        return False


class Event:
    """Thread synchronization event used for signaling."""

    _event: threading.Event

    @staticmethod
    def new() -> Event:
        """Create a new `Event`.

        Returns:
            Event: New event instance.
        """
        return Event()

    def __init__(self) -> None:
        """Initialize the event."""
        self._event = threading.Event()

    def wait(self) -> None:
        """Block until the event is notified."""
        self._event.wait()

    def notify(self) -> None:
        """Notify all waiting threads."""
        self._event.set()
