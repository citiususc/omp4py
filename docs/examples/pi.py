from omp4py import *


@omp
def pi(n: int) -> float:
    width = 1.0 / n
    total = 0.0

    with omp("parallel for reduction(+:total) num_threads(4)"):
        for i in range(n):
            x = (i + 0.5) * width
            total += 4.0 / (1.0 + x * x)

    return total * width


print(pi(1_000_000))
