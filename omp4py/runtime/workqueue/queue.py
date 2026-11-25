"""Queue of ready tasks of a thread.

Each thread of a team owns one `TaskQueue`. The owner pushes the tasks it creates at the tail and takes them back from
the tail (last in, first out), which keeps the data of the most recent task hot and limits the number of tasks that are
alive. The other threads steal from the head (first in, first out): the oldest tasks are the biggest ones, so a thief
takes few tasks and spends a lot of time on each of them.

The queue is a ring (a `list` whose size is a power of two) split in two zones by an index:

    head            split             tail
      |  shared zone  |  private zone  |
      oldest tasks   ->   newest tasks

- The private zone belongs to the owner: `push` and `pop` work on it with no mutex and no atomic operation, so a
  thread that creates tasks is not slowed down by the threads that steal from it.
- The shared zone is the one that thieves steal from, with the mutex of the queue taken. The owner only takes the
  mutex to move the split: to make tasks visible (`_release`, when the shared zone is empty and the private one has at
  least two tasks, it moves the older half) or to take back a task of the shared zone when its private zone is empty.

The number of tasks of the shared zone is kept in an atomic, so that a thread can see that there is nothing to steal
without taking the mutex. A thief that finds the mutex busy does not wait for it: it gives up and goes to another
queue.

The ring starts small and doubles when it is full, up to a maximum. When the maximum is reached `push` returns `False`
and the caller has to execute the task at once instead of queueing it, as the throttling of the LLVM OpenMP runtime
does.
"""

from __future__ import annotations

import typing

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.atomic import AtomicInt
from omp4py.runtime.lowlevel.mutex import Mutex

if typing.TYPE_CHECKING:
    from omp4py.runtime.lowlevel.numeric import pyint

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["TaskQueue"]


def _slot_put(queue: TaskQueue, i: pyint, task: object) -> None:
    """Write the slot of an index.

    The compiled version is in the `.pxd`.

    Args:
        queue (TaskQueue): Queue to write in.
        i (pyint): Index of the slot.
        task (object): Task to write.
    """
    queue._items[i & queue._mask] = task  # noqa: SLF001


def _slot_take(queue: TaskQueue, i: pyint) -> object:
    """Read the slot of an index and leave it empty.

    The compiled version is in the `.pxd`.

    Args:
        queue (TaskQueue): Queue to read from.
        i (pyint): Index of the slot.

    Returns:
        object: The task that was in the slot.
    """
    i &= queue._mask  # noqa: SLF001
    task = queue._items[i]  # noqa: SLF001
    queue._items[i] = None  # noqa: SLF001
    return task


# END_CYTHON_IGNORE


class TaskQueue:
    """Ring of tasks of one thread.

    Attributes:
        _items (list): The ring. Its length is a power of two, the slot of an index is `index & _mask`.
        _mask (pyint): Length of the ring minus one. Only the owner changes it, with the mutex taken.
        _max_size (pyint): Number of tasks from which `push` refuses more.
        _head (AtomicInt): Index of the oldest task. Changes with the mutex taken, and the owner reads it without it.
            It is an object of its own, so that the thieves that write it do not share a cache line with the fields
            that the owner writes at every push.
        _split (pyint): Index where the private zone starts (`_head <= _split <= _tail`). Only the owner changes it,
            with the mutex taken.
        _tail (pyint): Index after the newest task. Only the owner uses it.
        _head_cache (pyint): Last value of `_head` seen by the owner. The head only grows, so it can be older than
            the true one: the owner reads the real one only when the ring looks full.
        _shared (AtomicInt): Number of tasks of the shared zone (`_split - _head`). Changes with the mutex taken, but
            can be read without it.
        _lock (Mutex): Protects the shared zone, `_head`, `_split` and the ring when it grows.
    """

    _items: list[object]
    _mask: pyint
    _max_size: pyint
    _head: AtomicInt
    _split: pyint
    _tail: pyint
    _head_cache: pyint
    _shared: AtomicInt
    _lock: Mutex

    @staticmethod
    def new(initial: pyint, maximum: pyint) -> TaskQueue:
        """Create an empty queue.

        Args:
            initial (pyint): Length of the ring at the start. Rounded up to a power of two.
            maximum (pyint): Most tasks that the queue holds. Rounded up to a power of two. When the queue holds
                that many, `push` returns `False`.

        Returns:
            TaskQueue: Empty queue.
        """
        obj: TaskQueue = TaskQueue.__new__(TaskQueue)
        length: pyint = 2
        while length < initial:
            length <<= 1
        limit: pyint = length
        while limit < maximum:
            limit <<= 1
        obj._items = [None] * length
        obj._mask = length - 1
        obj._max_size = limit
        obj._head = AtomicInt.new(0)
        obj._split = 0
        obj._tail = 0
        obj._head_cache = 0
        obj._shared = AtomicInt.new(0)
        obj._lock = Mutex.new()
        return obj

    def push(self, task: object) -> bool:
        """Put a task at the tail. Only the owner of the queue calls it.

        Args:
            task (object): Task to queue. It cannot be `None`.

        Returns:
            bool: `False` if the queue is full and has reached its maximum; the task has not been queued and the
                caller has to execute it.
        """
        tail: pyint = self._tail
        if tail - self._head_cache > self._mask and not self._make_room(tail):  # looks full
            if self._shared.get() == 0 and tail - self._split > 1:  # the caller runs the task: feed the thieves
                self._release()
            return False
        _slot_put(self, tail, task)
        tail += 1
        self._tail = tail
        if self._shared.get() == 0 and tail - self._split > 1:
            self._release()
        return True

    def pop(self) -> object:
        """Take the newest task from the tail. Only the owner of the queue calls it.

        Returns:
            object: The task, or `None` if the queue is empty.
        """
        tail: pyint = self._tail
        if tail > self._split:  # the private zone has tasks
            tail -= 1
            task: object = _slot_take(self, tail)
            self._tail = tail
            if self._shared.get() == 0 and tail - self._split > 1:
                self._release()
            return task
        if self._shared.get() == 0:
            return None
        self._lock.lock()
        split: pyint = self._split
        head: pyint = self._head.get()
        if split == head:  # the thieves have taken all of them
            self._lock.unlock()
            return None
        split -= 1
        task = _slot_take(self, split)
        self._split = split
        self._tail = split  # the private zone was empty
        self._shared.set(split - head)
        self._lock.unlock()
        return task

    def steal(self) -> object:
        """Take the oldest task of the shared zone. Any thread calls it.

        It does not wait: if the owner or another thief is using the queue, it returns at once.

        Returns:
            object: The task, or `None` if there is nothing to steal or the queue is busy.
        """
        if self._shared.get() == 0 or not self._lock.test():
            return None
        head: pyint = self._head.get()
        split: pyint = self._split
        if head == split:  # the owner has taken the last one
            self._lock.unlock()
            return None
        task: object = _slot_take(self, head)
        head += 1
        self._head.set(head)
        self._shared.set(split - head)
        self._lock.unlock()
        return task

    def size(self) -> pyint:
        """Tell how many tasks the queue holds. Only the owner of the queue calls it.

        Returns:
            pyint: Number of tasks of both zones (the thieves can take some at the same time).
        """
        return self._tail - self._head.get()

    def stealable(self) -> pyint:
        """Tell how many tasks a thief can see, without taking the mutex.

        Returns:
            pyint: Number of tasks of the shared zone. By the time the caller uses it, it may have changed.
        """
        return self._shared.get()

    def _release(self) -> None:
        """Move the older half of the private zone to the shared zone.

        The private zone has to have two tasks or more.
        """
        self._lock.lock()
        split: pyint = self._split + ((self._tail - self._split) >> 1)
        self._split = split
        self._shared.set(split - self._head.get())
        self._lock.unlock()

    def _make_room(self, tail: pyint) -> bool:
        """Make a place for a task if the ring looks full: read the true head and, if it is needed, double the ring.

        Args:
            tail (pyint): Index where the task will be written.

        Returns:
            bool: `False` if the queue holds the maximum number of tasks. The mutex is not taken for that: the head
                only grows, so the queue cannot be less full than the owner sees.
        """
        head: pyint = self._head.get()
        self._head_cache = head
        if tail - head <= self._mask:
            return True
        if tail - head >= self._max_size:
            return False
        self._lock.lock()
        try:
            self._grow()
        except BaseException:
            self._lock.unlock()
            raise
        self._lock.unlock()
        return True

    def _grow(self) -> None:
        """Double the length of the ring.

        The mutex has to be taken and the ring has to be full.
        """
        items: list[object] = self._items
        new_mask: pyint = 2 * self._mask + 1
        new_items: list[object] = [None] * (new_mask + 1)
        i: pyint
        for i in range(self._head.get(), self._tail):
            new_items[i & new_mask] = items[i & self._mask]
        self._items = new_items
        self._mask = new_mask
