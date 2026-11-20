from omp4py import *


@omp
def only_some_threads():
    with omp("parallel num_threads(3)"):
        with omp("master"):
            print("master runs on thread", omp_get_thread_num())
        with omp("masked filter(2)"):
            print("masked runs on thread", omp_get_thread_num())


only_some_threads()
