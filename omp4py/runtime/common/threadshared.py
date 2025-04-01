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


class TaskQueue:
    pass


class SharedFactory:
    _context: _SharedEntry
    lock_mutex: lock.Mutex
    lock_barrier: lock.Barrier

    def __init__(self, cvars: controlvars.ControlVars):
        self._context = _SharedEntry.__new__(_SharedEntry)
        self._context.tp = 0
        self._context.obj = None
        self._context.next = atomic.AtomicObject.new()
        self.lock_mutex = lock.Mutex.new()
        self.lock_barrier = lock.Barrier.new(cvars.dataenv.team_size)

    def context(self) -> SharedContext:
        sh: SharedContext = SharedContext()
        sh._head = self._context
        sh._tail = self._context

        return sh

    def task_queue(self) -> TaskQueue:
        pass
