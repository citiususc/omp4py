from omp4py import *


def count_evens(n):                      # no @omp needed: the whole package is preprocessed
    total = 0
    with omp("parallel for reduction(+:total) num_threads(4)"):
        for i in range(n):
            if i % 2 == 0:
                total += 1
    return total
