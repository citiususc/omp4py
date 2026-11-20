from omp4py import *

counter = 0


@omp
def per_thread():
    global counter
    omp("threadprivate(counter)")
    counter = 100
    with omp("parallel num_threads(3) copyin(counter)"):
        counter += omp_get_thread_num()
        with omp("critical"):
            print("thread", omp_get_thread_num(), "counter", counter)
    return counter


print("master copy afterwards:", per_thread())
