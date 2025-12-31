import math
import time
from omp4py_examples.omputils import njit, pyomp, omp, use_pyomp, use_compiled, use_compiled_types

@njit
def _pyomp_pi(n):
    w = 1.0 / n
    PI = 0.0
    with pyomp("parallel for reduction(+:PI)"):
        for i in range(n):
            local = (i + 0.5) * w
            PI += 4.0 / (1.0 + local * local)
    return PI * w


@omp(compile=use_compiled())
def _omp4py_pi(n):
    w = 1.0 / n
    PI = 0.0
    with omp("parallel for reduction(+:PI)"):
        for i in range(n):
            local = (i + 0.5) * w
            PI += 4.0 / (1.0 + local * local)
    return PI * w


@omp(compile=use_compiled())
def _omp4py_pi_types(n: int):
    w: float = 1.0 / n
    PI: float = 0.0
    with omp("parallel for reduction(+:PI)"):
        for i in range(n):
            local: float = (i + 0.5) * w
            PI += 4.0 / (1.0 + local * local)
    return PI * w


def pi(n=2000000000):
    print(f"pi: n={n}")

    wtime = time.perf_counter()
    if use_pyomp():
        PI = _pyomp_pi(n)
    elif use_compiled_types():
        PI = _omp4py_pi_types(n)
    else:
        PI = _omp4py_pi(n)
    wtime = time.perf_counter() - wtime

    error = math.fabs(PI - math.pi)
    print(f"  Estimate       : {round(PI, 16)}")
    print(f"  Error          : {round(error, 16)}")
    print("Elapsed time      : %.6f" % wtime)
