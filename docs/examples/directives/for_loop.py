from omp4py import *


@omp
def squares(n):
    result = [0] * n
    with omp("parallel"):
        with omp("for"):
            for i in range(n):
                result[i] = i * i
    return result


print(squares(8))
