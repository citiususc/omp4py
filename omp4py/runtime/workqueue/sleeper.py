"""Sleeping threads that can be woken up one by one.

A thread that has no work and does not want to spin any longer has to sleep, and the thread that creates work has to
be able to wake up one sleeping thread (not all of them) at a low cost. This module keeps, for each thread of a team,
a slot where the thread publishes the event it sleeps on, and a counter of the threads that are sleeping.

A thread that wants to sleep follows the protocol:

1. `prepare` counts it as sleeping and publishes its event.
2. The thread looks again for the thing that it waits for (work, the end of a barrier...). The look has to be after
   `prepare`: with the operations of this module (all of them sequentially consistent), either the thread sees what
   was done before the wake-up, or the thread that wakes up sees the published event. A wake-up is not lost.
3. If there is something to do, `cancel` takes the event back. Otherwise, `wait` sleeps.

The thread that creates work does it first, and then calls `wake_one` or `wake_all`. When nobody sleeps, that costs
one atomic read.
"""

from __future__ import annotations

import typing

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.atomic import AtomicInt, AtomicObject
from omp4py.runtime.lowlevel.mutex import Event

if typing.TYPE_CHECKING:
    from omp4py.runtime.lowlevel.numeric import pyint

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["Sleepers"]


def _slot_set(sleepers: Sleepers, i: pyint, event: Event) -> None:
    """Publish an event in the slot of a thread.

    The compiled version is in the `.pxd`.

    Args:
        sleepers (Sleepers): Slots of the team.
        i (pyint): Number of the thread.
        event (Event): Event that the thread sleeps on.
    """
    sleepers._slots[i].set(event)  # noqa: SLF001


def _slot_take(sleepers: Sleepers, i: pyint) -> Event | None:
    """Take the event of the slot of a thread, if any.

    The compiled version is in the `.pxd`.

    Args:
        sleepers (Sleepers): Slots of the team.
        i (pyint): Number of the thread.

    Returns:
        Event | None: The event, or `None` if the thread has not published one or it was taken already.
    """
    return sleepers._slots[i].exchange(None)  # noqa: SLF001


# END_CYTHON_IGNORE


class Sleepers:
    """Sleep slots of the threads of a team.

    Attributes:
        _slots (list[AtomicObject]): For each thread, the event it sleeps on, or `None`.
        _sleeping (AtomicInt): Number of threads between `prepare` and the end of `wait` or `cancel`.
        _n (pyint): Number of threads of the team.
    """

    _slots: list[AtomicObject]
    _sleeping: AtomicInt
    _n: pyint

    @staticmethod
    def new(n: pyint) -> Sleepers:
        """Create the slots of the `n` threads of a team.

        Args:
            n (pyint): Number of threads of the team.

        Returns:
            Sleepers: Slots with no sleeping thread.
        """
        obj: Sleepers = Sleepers.__new__(Sleepers)
        obj._slots = [AtomicObject.new() for _ in range(n)]
        obj._sleeping = AtomicInt.new(0)
        obj._n = n
        return obj

    def prepare(self, tid: pyint) -> Event:
        """Count a thread as sleeping and publish the event it will sleep on.

        Args:
            tid (pyint): Number of the thread.

        Returns:
            Event: Event to give to `wait`.
        """
        self._sleeping.add(1)  # first, so a thread that wakes up and reads the counter does not miss this one
        event: Event = Event.new()
        _slot_set(self, tid, event)
        return event

    def cancel(self, tid: pyint) -> bool:
        """Take back the event published by `prepare`, because the thread does not sleep.

        Args:
            tid (pyint): Number of the thread.

        Returns:
            bool: `True` if a thread that wakes up had already taken the event, that is, the wake-up was used for a
                thread that did not sleep. The caller can pass it on with `wake_one`.
        """
        taken: bool = _slot_take(self, tid) is None
        self._sleeping.sub(1)
        return taken

    def wait(self, event: Event) -> None:
        """Sleep until the event is notified, which can be already.

        Args:
            event (Event): Event returned by `prepare`.
        """
        event.wait()
        self._sleeping.sub(1)

    def wake_one(self, tid: pyint) -> bool:
        """Wake up one sleeping thread.

        The search starts after the thread that wakes up, so the same thread is not always the one woken.

        Args:
            tid (pyint): Number of the thread that wakes up.

        Returns:
            bool: `True` if a thread was woken up.
        """
        if self._sleeping.get() == 0:
            return False
        n: pyint = self._n
        i: pyint
        event: Event | None
        for i in range(1, n + 1):
            event = _slot_take(self, (tid + i) % n)
            if event is not None:
                event.notify()
                return True
        return False

    def wake_all(self) -> pyint:
        """Wake up all the sleeping threads.

        Returns:
            pyint: Number of threads that were woken up.
        """
        if self._sleeping.get() == 0:
            return 0
        woken: pyint = 0
        i: pyint
        event: Event | None
        for i in range(self._n):
            event = _slot_take(self, i)
            if event is not None:
                event.notify()
                woken += 1
        return woken

    def any_sleeping(self) -> bool:
        """Tell whether some thread is sleeping or about to sleep.

        Returns:
            bool: `True` if the counter of sleeping threads is not zero.
        """
        return self._sleeping.get() > 0
