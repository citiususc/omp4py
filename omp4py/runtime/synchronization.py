from omp4py.runtime.common import thread
from omp4py.runtime.common.barrier import task_barrier

def sync_barrier():
    task_barrier(thread.current().task)


def mutex_lock():
    thread.current().parallel.lock_mutex.lock()


def mutex_unlock():
    thread.current().parallel.lock_mutex.unlock()

