from omp4py import *


@omp
def offload(a, b, n):
    # Valid OpenMP 6.0 syntax: it is parsed, but the transformation of `target` is not available yet.
    with omp("target teams distribute parallel for map(tofrom: a, b)"):
        for i in range(n):
            b[i] = 2 * a[i]


offload([1, 2, 3], [0, 0, 0], 3)
