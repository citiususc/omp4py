"""Base task and synchronization primitives for the `omp4py` runtime.

This module defines the fundamental execution unit (`Task`) used to define
all runtime constructs in the OpenMP-like execution model implemented by
`omp4py`.

It also provides shared execution state management (`SharedContext`)
and synchronization primitives (`Barrier`) used to coordinate thread
teams during parallel regions.
"""

from __future__ import annotations

import os
import sys
import typing

import cython

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.atomic import AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.mutex import Event

if typing.TYPE_CHECKING:
    from omp4py.runtime.icvs import Data
    from omp4py.runtime.lowlevel.numeric import pyint



# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["SharedContext", "Task", "instanceof", "same_class"]


def same_class(obj1: object, obj2: object) -> bool:
    """Check whether two objects belong to the same runtime class.

    This function is used to validate that all threads execute consistent
    structured blocks, as required by the OpenMP execution model.

    Args:
        obj1 (object): First object.
        obj2 (object): Second object.

    Returns:
        bool: True if both objects are instances of the same class.
    """
    return obj1.__class__ == obj2.__class__

def instanceof(obj: object, cls: type[object]) -> bool:
    """Check whether an object belongs to a class.

    This function is used to validate that all threads execute consistent
    structured blocks, as required by the OpenMP execution model.

    Args:
        obj (object): The object to check.
        cls (type[object]): The class to compare against.

    Returns:
        bool: True if the object is an instance of the class.
    """
    return isinstance(obj, cls)

# END_CYTHON_IGNORE


class Task:
    """Base execution unit for all runtime tasks in `omp4py`.

    A `Task` represents the fundamental unit of execution in the runtime.
    All higher-level constructs (e.g., parallel regions) are built on top
    of this abstraction.

    Attributes:
        _return_to (Task | None): Previous task to restore after execution.
        shared (SharedContext): Shared execution state across threads.
        icvs (Data): Internal control variables for runtime behavior.
        barrier (Barrier): Synchronization barrier for the task team.
    """

    _return_to: Task | None
    shared: SharedContext
    icvs: Data
    barrier: Barrier

    def _new_task(self, shared: SharedContext, icvs: Data, barrier: Barrier) -> Task:
        """Initialize a new task instance.

        This method sets up the execution context for a task, including
        shared state, internal control variables, and a team barrier
        based on the current team size.

        Args:
            shared (SharedContext): Shared execution context.
            icvs (Data): Internal control variables.
            barrier (Barrier): Synchronization barrier for the task.

        Returns:
            Task: Initialized task instance.
        """
        self._return_to = None
        self.shared = shared
        self.icvs = icvs
        self.barrier = barrier
        return self


class SharedItem:
    """Node in a lock-free shared state chain.

    Each `SharedItem` represents a value in a linked structure used by
    `SharedContext` to propagate shared data across threads in a
    deterministic manner.

    Attributes:
        value (object): Stored shared value.
        next (AtomicObject[SharedItem]): Atomic pointer to next node.
    """
    value: object
    next: AtomicObject[SharedItem]

    @staticmethod
    def new(value: object) -> SharedItem:
        """Create a new shared item node.

        Args:
            value (object): Value to store.

        Returns:
            SharedItem: Newly created shared node.
        """
        obj: SharedItem = SharedItem.__new__(SharedItem)
        obj.value = value
        obj.next = AtomicObject.new()
        return obj


class SharedContext:
    """Lock-free shared execution context for runtime state propagation.

    This structure provides a deterministic mechanism to synchronize
    shared values across threads in a parallel region.

    Attributes:
        _current (AtomicObject[SharedItem]): Pointer to current shared value.
    """
    _current: AtomicObject[SharedItem]

    @staticmethod
    def new() -> SharedContext:
        """Create a new shared context.

        Returns:
            SharedContext: Fresh shared execution context.
        """
        obj: SharedContext = SharedContext.__new__(SharedContext)
        obj._current = SharedItem.new(None).next
        return obj

    def mirror(self) -> SharedContext:
        """Create a new SharedContext referencing the same underlying shared state.

        This method is used to initialize per-thread contexts that share the same
        global state, but maintain independent progression through the internal
        shared chain.

        Returns:
            SharedContext: A new context instance sharing the same underlying
            shared state.
        """
        obj: SharedContext = SharedContext.__new__(SharedContext)
        obj._current = self._current
        return obj

    def get(self, cls: type[object]) -> object:
        """Retrieve a shared value across threads.

        This method returns the canonical shared object if it has already been
        initialized and matches the expected class.

        It is designed to be fast in the common case where the shared value
        is already available, avoiding unnecessary synchronization or creation.

        If the shared value does not exist, is invalid, or does not match the
        expected class, `None` is returned. In that case, the caller is expected
        to fall back to `sync()` to initialize or synchronize the value.

        Args:
            cls (type[object]): Expected class of the shared value.

        Returns:
            object | None: The shared canonical value if it exists and matches
            the expected class; otherwise `None`.
        """
        shared : SharedItem | None =self._current.get()
        if shared is None:
            return None
        stored = cython.cast(SharedItem, self._current.get())
        self._current = stored.next
        if not instanceof(stored.value, cls):
            return None # Simulate an empty value to delegate to the sync function
        return stored.value

    def sync(self, obj: object) -> object:
        """Synchronize a shared value across threads.

        This method ensures that all threads agree on the same value for
        a shared object. The first thread to set the value defines it,
        and subsequent threads must observe a consistent type/class.

        If inconsistency is detected, the runtime aborts execution.

        Args:
            obj (object): Proposed shared value.

        Returns:
            object: The canonical shared value.

        Raises:
            SystemExit: If inconsistent execution is detected across threads.
        """
        if self._current.get() is None and self._current.set(shared := SharedItem.new(obj)):
            self._current = shared.next
            return obj
        stored = cython.cast(SharedItem, self._current.get())
        self._current = stored.next
        if not same_class(obj, stored.value):
            print(  # noqa: T201
                "error: inconsistent directive execution detected\n"
                "All threads must follow the same structured block according to the OpenMP specification.\n"
                "Aborting program.",
                file=sys.stderr,
            )
            os._exit(1)
        return stored.value


class Barrier:
    """Reusable synchronization barrier for thread teams.

    This barrier implements a phase-based synchronization mechanism where
    all threads in a team must reach the barrier before any can proceed.

    It supports reuse across multiple synchronization cycles using a
    generation counter.

    Attributes:
        _parties (pyint): Number of participating threads.
        _waiting (AtomicInt): Current number of waiting threads.
        _event (AtomicObject[Event]): Event used for blocking/wakeup.
        _gen (pyint): Barrier generation counter.
    """
    _parties: pyint
    _waiting: AtomicInt
    _event: AtomicObject[Event]
    _gen: pyint

    @staticmethod
    def new(parties: pyint) -> Barrier:
        """Create a new barrier.

        Args:
            parties (pyint): Number of threads in the team.

        Returns:
            Barrier: Initialized barrier instance.
        """
        obj: Barrier = Barrier.__new__(Barrier)
        obj._parties = parties
        obj._waiting = AtomicInt.new(0)
        obj._event = AtomicObject.new()
        obj._event.set(Event.new())
        obj._gen = 0
        return obj

    def wait(self) -> bool:
        """Wait for all threads to reach the barrier.

        Returns:
            bool: True if the barrier completed a full synchronization
            cycle during this call. False if the barrier was interrupted.
        """
        my_gen = self._gen
        event = cython.cast(Event, self._event.get())
        if self._waiting.add(1) == self._parties:
            self._waiting.set(0)
            self._gen += 1
            self.interrupt()
            return True
        event.wait()
        return my_gen < self._gen

    def interrupt(self) -> None:
        """Interrupt all threads waiting on the barrier.

        This forces all waiting threads to wake up and replaces the
        underlying event object.
        """
        new_event = Event.new()
        old_event: Event = cython.cast(Event, self._event.exchange(new_event))
        old_event.notify()
