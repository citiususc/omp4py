from omp4py.runtime.basics.threadlocal cimport thread_id

from omp4py.runtime.basics.types cimport *

cdef class Mutex:

    @staticmethod
    cdef Mutex new():
        m: Mutex = Mutex.__new__(Mutex)
        m._mutex = PyMutex(0)
        return m

    cdef void lock(self):
        PyMutex_Lock(&self._mutex)

    cdef void unlock(self):
        PyMutex_Unlock(&self._mutex)

    cdef bint test(self):
        return PyMutex_LockFast(&self._mutex)

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()
        return False

cdef class RMutex:

    @staticmethod
    cdef RMutex new():
        m: RMutex = RMutex.__new__(RMutex)
        m._mutex = PyMutex(0)
        m._own = -1
        m._level = 0
        return m

    cdef void lock(self):
        if self._own == -1:
            self._own = thread_id()
            self._level += 1
            PyMutex_Lock(&self._mutex)
        elif self._own != thread_id():
            self._level += 1

    cdef void unlock(self):
        if self._own == thread_id():
            self._level -= 1
            if self._level == 0:
                PyMutex_Unlock(&self._mutex)

    cdef bint test(self):
        if self._own == -1:
            result: int = PyMutex_LockFast(&self._mutex)
            if result:
                self._level += 1
            return result
        elif self._own != thread_id():
            self._level += 1
            return True

    def __enter__(self):
        self.lock()

    def __exit__(self, exc_type, exc_value, traceback):
        self.unlock()
        return False


cdef class Event:

    @staticmethod
    cdef Event new():
        e: Event = Event.__new__(Event)
        e._event = PyEvent(0)
        return e

    cdef void wait(self):
        PyEvent_Wait(&self._event)

    cdef void notify(self):
        _PyEvent_Notify(&self._event)
