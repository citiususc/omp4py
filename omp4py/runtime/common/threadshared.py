import typing
# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
from omp4py.runtime.basics import atomic, lock
from omp4py.runtime.common import controlvars
from omp4py.runtime.basics.types import *
# END_CYTHON_IMPORTS


from cython import cast


class _SharedEntry:
    obj: typing.Any
    tp: pyint
    next: atomic.AtomicObject


class SharedContext:
    _tail: _SharedEntry

    @staticmethod
    def new() -> 'SharedContext':
        sc: SharedContext = SharedContext.__new__(SharedContext)
        sc._tail = _SharedEntry.__new__(_SharedEntry)
        sc._tail.tp = -1
        sc._tail.obj = None
        sc._tail.next = atomic.AtomicObject.new()

        return sc

    def push(self, tp: pyint, value: typing.Any) -> typing.Any:
        entry: _SharedEntry = _SharedEntry.__new__(_SharedEntry)
        entry.tp = tp
        entry.obj = value
        entry.next = atomic.AtomicObject.new()
        if self._tail.next.set(entry):
            self._tail = entry
        else:
            self._tail = cast(_SharedEntry, self._tail.next.get())
            if self._tail.tp != tp:
                raise ValueError(f'Each thread will execute the same instruction stream {self._tail.tp} != {tp}')
        return self._tail.obj

    def get(self, tp: pyint) -> typing.Any:
        next: _SharedEntry = self._tail
        while next is not None:
            if next.tp == tp:
                return next.obj
            next = cast(_SharedEntry, next.next.get())
        return None

    def move_last(self):
        next: _SharedEntry = self._tail
        while next is not None:
            self._tail = next
            next = cast(_SharedEntry, self._tail.next.get())

    def __copy__(self) -> 'SharedContext':
        sc: SharedContext = SharedContext.__new__(SharedContext)
        sc._tail = self._tail

        return sc


class _QueueEntry:
    obj: typing.Any
    free: atomic.AtomicFlag
    next: atomic.AtomicObject


class TaskQueue:
    _head: _QueueEntry
    _tail: _QueueEntry
    _history: _QueueEntry | None

    @staticmethod
    def new() -> 'TaskQueue':
        tq: TaskQueue = TaskQueue.__new__(TaskQueue)

        tq._head = tq._tail = _QueueEntry.__new__(_QueueEntry)
        tq._head.free = atomic.AtomicFlag.new()
        tq._head.free.test_and_set()
        tq._head.obj = None
        tq._head.next = atomic.AtomicObject.new()
        tq._history = None

        return tq

    def add(self, value: typing.Any) -> None:
        entry: _QueueEntry = _QueueEntry.__new__(_QueueEntry)
        entry.obj = value
        entry.free = atomic.AtomicFlag.new()
        entry.next = atomic.AtomicObject.new()
        while not self._tail.next.set(entry):
            self._tail = cast(_QueueEntry, self._tail.next.get())

    def take(self) -> typing.Any:
        entry = cast(_QueueEntry, self._head.next.get())
        while entry is not None and entry.free.no_clear_test_and_set():
            entry = cast(_QueueEntry, entry.next.get())

        if entry is not None:
            self._head = entry
            return entry.obj

    def set_history(self) -> None:
        self._history = self._head

    def history_take(self) -> typing.Any:
        if self._history is self._head:
            self._history = None
        if self._history is None:
            return None
        obj: typing.Any = self._history.obj
        self._history = cast(_QueueEntry, self._history.next.get())
        return obj

    def __copy__(self) -> 'TaskQueue':
        tq: TaskQueue = TaskQueue.__new__(TaskQueue)
        tq._head = self._head
        tq._tail = self._tail

        return tq


class SharedFactory:
    _context: SharedContext
    _tasks: TaskQueue
    lock_mutex: lock.Mutex

    @staticmethod
    def new(cvars: controlvars.ControlVars):
        sf: SharedFactory = SharedFactory.__new__(SharedFactory)

        sf._context = SharedContext.new()
        sf._tasks = TaskQueue.new()
        sf.lock_mutex = lock.Mutex.new()

        return sf

    def context(self) -> SharedContext:
        return self._context.__copy__()

    def task_queue(self) -> TaskQueue:
        return self._tasks.__copy__()
