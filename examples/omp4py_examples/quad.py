import math
import time
from omp4py_examples.omputils import njit, pyomp, omp, use_pyomp, use_compiled, use_compiled_types

@njit
def _f(x):
    return 50.0 / (math.pi * (2500.0 * x * x + 1.0))


@njit
def _pyomp_quad(n, a, b):
    total = 0.0

    with pyomp("parallel for reduction(+:total)"):
        for i in range(n):
            x = ((n - i - 1) * a + i * b) / (n - 1)
            total = total + _f(x)

    return total


@omp(compile=use_compiled())
def _omp4py_quad(n, a, b):
    total = 0.0

    with omp("parallel for reduction(+:total)"):
        for i in range(n):
            x = ((n - i - 1) * a + i * b) / (n - 1)
            total = total + _f(x)

    return total


@omp(compile=use_compiled())
def _omp4py_quad_types(n: int, a: float, b: float):
    total: float = 0.0
    PI: float = math.pi

    with omp("parallel for reduction(+:total)"):
        for i in range(n):
            x: float = ((n - i - 1) * a + i * b) / (n - 1)
            total = total + 50.0 / (PI * (2500.0 * x * x + 1.0))

    return total


def quad(n=1000000000):
    print(f"quad: n={n}")
    a = 0.0
    b = 10.0
    exact = 0.49936338107645674464

    print("  Estimate the integral of f(x) from A to B.\n" +
          "  f(x) = 50 / ( pi * ( 2500 * x * x + 1 ) ).\n" +
          "\n" +
          f"  A        = {a}\n" +
          f"  B        = {b}\n" +
          f"  N        = {n}\n" +
          f"  Exact    = {exact}")

    wtime = time.perf_counter()
    if use_pyomp():
        total = _pyomp_quad(n, a, b)
    elif use_compiled_types():
        total = _omp4py_quad_types(n, a, b)
    else:
        total = _omp4py_quad(n, a, b)
    wtime = time.perf_counter() - wtime

    total = (b - a) * total / n
    error = math.fabs(total - exact)
    print(f"  Estimate       : {round(total, 16)}")
    print(f"  Error          : {round(error, 16)}")
    print("Elapsed time      : %.6f" % wtime)
