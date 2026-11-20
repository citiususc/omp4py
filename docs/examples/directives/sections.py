from omp4py import *


@omp
def two_jobs():
    result = {}
    with omp("parallel sections num_threads(2)"):
        with omp("section"):
            result["a"] = sum(range(1000))
        with omp("section"):
            result["b"] = sum(range(2000))
    return result


print(two_jobs())
