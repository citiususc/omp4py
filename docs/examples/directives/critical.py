from omp4py import *


@omp
def counters():
    total = 0
    named = 0
    with omp("parallel num_threads(4)"):
        for _ in range(1000):
            with omp("critical"):
                total += 1
            with omp("critical(other)"):
                named += 2
    return total, named


print(counters())
