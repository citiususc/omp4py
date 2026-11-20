from omp4py import *


@omp
def hello():
    with omp("parallel num_threads(4)"):
        print("hello from thread", omp_get_thread_num(), "of", omp_get_num_threads())


hello()
