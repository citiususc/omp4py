import typing
from omp4py.runtime.basics import atomic, lock
from omp4py.runtime.common import controlvars
from omp4py.runtime.basics.types import *
from omp4py.runtime.basics.casting import *


class _SharedEntry:
    obj: typing.Any
    tp: pyint
    next: atomic.AtomicObject


class SharedContext:
    _head: _SharedEntry
    _tail: _SharedEntry

    def push(self, tp: pyint, value: typing.Any) -> typing.Any:
        entry: _SharedEntry = _SharedEntry.__new__(_SharedEntry)
        entry.tp = tp
        entry.obj = value
        entry.next = atomic.AtomicObject.new()
        if not self._tail.next.set(entry):
            entry = cast(_SharedEntry, self._tail.next.get())
            if entry.tp != tp:
                raise ValueError('Each thread will execute the same instruction stream')
        self._tail = entry
        return entry.obj

    def pop(self) -> None:
        self._head = cast(_SharedEntry, self._head.next.get())

    def has(self) -> bool:
        return self._head.next.get() is not None


class _QueueEntry:
    obj: typing.Any
    free: atomic.AtomicFlag
    next: atomic.AtomicObject


class TaskQueue:
    _head: _QueueEntry
    _tail: _QueueEntry

    def add(self, value: typing.Any) -> None:
        entry: _QueueEntry = _QueueEntry.__new__(_QueueEntry)
        entry.obj = value
        entry.free = atomic.AtomicFlag.new()
        entry.next = atomic.AtomicObject.new()
        while not self._tail.next.set(entry):
            entry = cast(_QueueEntry, self._tail.next.get())
        self._tail = entry

    def take(self) -> typing.Any:
        entry = cast(_QueueEntry, self._head.next.get())
        while entry is not None and not entry.free.no_clear_test_and_set():
            entry = cast(_QueueEntry, entry.next.get())

        if entry is not None:
            self._head = entry
            return entry.obj


class SharedFactory:
    _context: _SharedEntry
    _tasks: _QueueEntry
    lock_mutex: lock.Mutex
    lock_barrier: lock.Barrier

    def __init__(self, cvars: controlvars.ControlVars):
        self._context = _SharedEntry.__new__(_SharedEntry)
        self._context.tp = 0
        self._context.obj = None
        self._context.next = atomic.AtomicObject.new()
        self._tasks = _QueueEntry.__new__(_QueueEntry)
        self._tasks.free = atomic.AtomicFlag.new()
        self._tasks.free.test_and_set()
        self._tasks.fast = False
        self._tasks.obj = None
        self._tasks.next = atomic.AtomicObject.new()
        self.lock_mutex = lock.Mutex.new()
        self.lock_barrier = lock.Barrier.new(cvars.dataenv.team_size)

    def context(self) -> SharedContext:
        sh: SharedContext = SharedContext.__new__(SharedContext)
        sh._head = self._context
        sh._tail = self._context

        return sh

    def task_queue(self) -> TaskQueue:
        tq: TaskQueue = TaskQueue.__new__(TaskQueue)
        tq._head = self._tasks
        tq._tail = self._tasks

        return tq
