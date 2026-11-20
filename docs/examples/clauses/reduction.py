from omp4py import *


@omp
def reductions(n):
    total = 0
    product = 1
    all_even = True
    with omp("parallel for reduction(+:total) reduction(*:product) reduction(and:all_even) num_threads(4)"):
        for i in range(1, n + 1):
            total += i
            product *= 2
            all_even = all_even and i % 2 == 0
    return total, product, all_even


print(reductions(5))
