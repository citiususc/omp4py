import random

from omp4py import *


@omp
def matmul(a, b, c, n):
    with omp("parallel for collapse(2) num_threads(4)"):
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += a[i][k] * b[k][j]
                c[i][j] = acc


n = 64
random.seed(0)
a = [[random.random() for _ in range(n)] for _ in range(n)]
b = [[random.random() for _ in range(n)] for _ in range(n)]
c = [[0.0] * n for _ in range(n)]

matmul(a, b, c, n)

expected = [[sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
print("max error:", max(abs(c[i][j] - expected[i][j]) for i in range(n) for j in range(n)))
