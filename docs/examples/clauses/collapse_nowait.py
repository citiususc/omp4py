from omp4py import *


@omp
def two_loops(n, m):
    count = 0
    with omp("parallel num_threads(3)"):
        with omp("for collapse(2) reduction(+:count) nowait"):
            for i in range(n):
                for j in range(m):
                    count += 1
        omp("barrier")
    return count


print(two_loops(4, 5))
