from omp4py.runtime.common import thread

def barrier():
    thread.current().parallel.lock_barrier.wait()


def mutex_lock():
    thread.current().parallel.lock_mutex.lock()


def mutex_unlock():
    thread.current().parallel.lock_mutex.unlock()

