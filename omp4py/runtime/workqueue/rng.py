"""Per-thread pseudo-random numbers used by the `omp4py` runtime.

The tasking runtime has to pick a victim to steal a task from, and it has to do it often and cheaply, so the
generator is a 32-bit `xorshift` (three shifts and three xors) whose state is private to a thread: there are no
atomics and no shared state. The state of all the threads of a team is kept together in one array.

The numbers are kept in 32 bits with an explicit mask, so the pure Python implementation and the compiled one give
the same sequence.
"""

from __future__ import annotations

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["Rng"]


def _slot(tid: pyint) -> pyint:
    """Give the place of the state of a thread in the array of states.

    The compiled version is in the `.pxd` and leaves a cache line between the states of two threads.

    Args:
        tid (pyint): Number of the thread.

    Returns:
        pyint: Place of the state of the thread.
    """
    return tid


def _length(n: pyint) -> pyint:
    """Give the length of the array of states of a team.

    The compiled version is in the `.pxd`.

    Args:
        n (pyint): Number of threads of the team.

    Returns:
        pyint: Length of the array.
    """
    return n


# END_CYTHON_IGNORE


class Rng:
    """State of the `xorshift` generator of every thread of a team.

    Attributes:
        _state (pyint_array): Current value of the generator of each thread, at `_slot(thread)`. The compiled version
            leaves a cache line between threads, so threads that write at the same time do not share it. Never zero.
    """

    _state: pyint_array

    @staticmethod
    def new(n: pyint, seed: pyint) -> Rng:
        """Create the generators of the `n` threads of a team.

        The state of every thread is different and different seeds give different sequences, so two threads do not
        pick the same victims in the same order.

        Args:
            n (pyint): Number of threads of the team.
            seed (pyint): Seed of the team.

        Returns:
            Rng: Generators ready to use.
        """
        mask: pyint = 0xFFFFFFFF
        obj: Rng = Rng.__new__(Rng)
        state: pyint_array = new_pyint_array(_length(n))
        i: pyint
        value: pyint
        for i in range(n):
            value = ((seed & mask) * 1103515245 + (i + 1) * 40503) & mask
            state[_slot(i)] = 1 if value == 0 else value
        obj._state = state
        return obj

    def next(self, tid: pyint) -> pyint:
        """Advance the generator of a thread.

        Args:
            tid (pyint): Number of the thread.

        Returns:
            pyint: Next value, between 1 and 2**32 - 1.
        """
        mask: pyint = 0xFFFFFFFF
        x: pyint = self._state[_slot(tid)]
        x ^= (x << 13) & mask
        x ^= x >> 17
        x ^= (x << 5) & mask
        self._state[_slot(tid)] = x
        return x

    def victim(self, tid: pyint, n: pyint) -> pyint:
        """Pick a random thread that is not the given one.

        There is no loop: the value is taken between 0 and `n - 2`, and the numbers from `tid` on are moved one place.

        Args:
            tid (pyint): Number of the thread that picks.
            n (pyint): Number of threads of the team.

        Returns:
            pyint: Number of a thread other than `tid`, or -1 if the team has only one thread.
        """
        if n < 2:  # noqa: PLR2004
            return -1
        v: pyint = self.next(tid) % (n - 1)
        return v + 1 if v >= tid else v
