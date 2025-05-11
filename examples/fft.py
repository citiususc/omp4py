import numpy as np
import math
import time
from omputils import njit, pyomp, omp, omp_pure, use_pyomp, use_pure, use_compiled, use_compiled_types

if use_pure():
    omp = omp_pure

try:
    import cython
except ImportError:
    pass


def _ggl(seed):
    d2 = 0.2147483647e10

    seed = math.fmod(16807.0 * seed, d2)
    value = ((seed - 1.0) / (d2 - 1.0))

    return value, seed


def _cffti(n, w):
    n2 = n // 2
    aw = 2.0 * math.pi / n

    for i in range(n2):
        arg = aw * i
        w[i * 2 + 0] = math.cos(arg)
        w[i * 2 + 1] = math.sin(arg)


def _cfft2(n, x, y, w, sgn, _step):
    m = math.log(n) / math.log(1.99)
    mj = 1

    tgle = 1
    _step(n, mj, x[0 * 2 + 0:], x[(n // 2) * 2 + 0:], y[0 * 2 + 0:], y[mj * 2 + 0:], w, sgn)

    if n == 2:
        return

    for j in range(int(m - 2)):
        mj = mj * 2
        if tgle:
            _step(n, mj, y[0 * 2 + 0:], y[(n // 2) * 2 + 0:], x[0 * 2 + 0:], x[mj * 2 + 0:], w, sgn)
            tgle = 0
        else:
            _step(n, mj, x[0 * 2 + 0:], x[(n // 2) * 2 + 0:], y[0 * 2 + 0:], y[mj * 2 + 0:], w, sgn)
            tgle = 1

    if tgle:
        y[:n] = x[:n]

    mj = n // 2
    _step(n, mj, x[0 * 2 + 0:], x[(n // 2) * 2 + 0:], y[0 * 2 + 0:], y[mj * 2 + 0:], w, sgn)


@njit
def _pyomp_step(n, mj, a, b, c, d, w, sgn):
    mj2 = 2 * mj
    lj = n // mj2

    with pyomp("parallel"):
        wjw = np.empty(2)
        with pyomp("for"):
            for j in range(lj):
                jw = j * mj
                ja = jw
                jb = ja
                jc = j * mj2
                jd = jc

                wjw[0] = w[jw * 2 + 0]
                wjw[1] = w[jw * 2 + 1]

                if sgn < 0.0:
                    wjw[1] = -wjw[1]

                for k in range(mj):
                    c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0]
                    c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1]

                    ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0]
                    ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1]

                    d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu
                    d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu


@omp(compile=use_compiled())
def _omp4py_step(n, mj, a, b, c, d, w, sgn):
    mj2 = 2 * mj
    lj = n // mj2

    with omp("parallel"):
        wjw = np.empty(2)
        with omp("for"):
            for j in range(lj):
                jw = j * mj
                ja = jw
                jb = ja
                jc = j * mj2
                jd = jc

                wjw[0] = w[jw * 2 + 0]
                wjw[1] = w[jw * 2 + 1]

                if sgn < 0.0:
                    wjw[1] = -wjw[1]

                for k in range(mj):
                    c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0]
                    c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1]

                    ambr = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0]
                    ambu = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1]

                    d[(jd + k) * 2 + 0] = wjw[0] * ambr - wjw[1] * ambu
                    d[(jd + k) * 2 + 1] = wjw[1] * ambr + wjw[0] * ambu


@omp(compile=use_compiled())
def _omp4py_step_types(n: int, mj: int, a2, b2, c2, d2, w2, sgn: float):
    mj2: int = 2 * mj
    lj: int = n // mj2

    a: cython.double[:] = a2
    b: cython.double[:] = b2
    c: cython.double[:] = c2
    d: cython.double[:] = d2
    w: cython.double[:] = w2

    with omp("parallel"):
        wjw0: float = 0
        wjw1: float = 1
        with omp("for"):
            for j in range(lj):
                jw: int = j * mj
                ja: int = jw
                jb: int = ja
                jc: int = j * mj2
                jd: int = jc

                wjw0 = w[jw * 2 + 0]
                wjw1 = w[jw * 2 + 1]

                if sgn < 0.0:
                    wjw1 = -wjw1

                k: int
                for k in range(mj):
                    c[(jc + k) * 2 + 0] = a[(ja + k) * 2 + 0] + b[(jb + k) * 2 + 0]
                    c[(jc + k) * 2 + 1] = a[(ja + k) * 2 + 1] + b[(jb + k) * 2 + 1]

                    ambr: float = a[(ja + k) * 2 + 0] - b[(jb + k) * 2 + 0]
                    ambu: float = a[(ja + k) * 2 + 1] - b[(jb + k) * 2 + 1]

                    d[(jd + k) * 2 + 0] = wjw0 * ambr - wjw1 * ambu
                    d[(jd + k) * 2 + 1] = wjw1 * ambr + wjw0 * ambu


def fft(ln2_max=22, nits=10000, seed=331):
    print(f"fft: ln2_max={ln2_max}, nits={nits}, seed={seed}")
    # https://people.math.sc.edu/Burkardt/cpp_src/fft_openmp/fft_openmp.html
    print("  Accuracy check:")
    print("            N      NITS    Error         Time          Time/Call     MFLOPS")

    wtotal = 0
    if use_pyomp():
        _step = _pyomp_step
    elif use_compiled_types():
        _step = _omp4py_step_types
    else:
        _step = _omp4py_step
    n = 1
    # LN2 is the log base 2 of N.  Each increase of LN2 doubles N.
    for ln2 in range(1, ln2_max + 1):
        n = 2 * n

        #  Allocate storage for the complex arrays W, X, Y, Z.

        #  We handle the complex arithmetic,
        #  and store a complex number as a pair of doubles, a complex vector as a doubly
        #  dimensioned array whose second dimension is 2.

        w = np.empty(n)
        x = np.empty(2 * n)
        y = np.empty(2 * n)
        z = np.empty(2 * n)

        first = 1

        for icase in range(2):
            if first:
                for i in range(0, 2 * n, 2):
                    z0, seed = _ggl(seed)
                    z1, seed = _ggl(seed)
                    x[i] = z0
                    z[i] = z0
                    x[i + 1] = z1
                    z[i + 1] = z1
            else:
                for i in range(0, 2 * n, 2):
                    z0 = 0.0
                    z1 = 0.0
                    x[i] = z0
                    z[i] = z0
                    x[i + 1] = z1
                    z[i + 1] = z1

            # Initialize the sine and cosine tables.
            _cffti(n, w)

            # Transform forward, back
            if first:
                sgn = 1.0
                _cfft2(n, x, y, w, sgn, _step)
                sgn = -1.0
                _cfft2(n, y, x, w, sgn, _step)

                # Results should be same as initial multiplied by N.
                fnm1 = 1.0 / n
                error = 0.0
                for i in range(0, 2 * n, 2):
                    error += math.pow(z[i] - fnm1 * x[i], 2) + math.pow(z[i + 1] - fnm1 * x[i + 1], 2)
                error = math.sqrt(fnm1 * error)
                print(f" {n:>12}  {nits:>8} {round(error, 2):>12}", end="")
                first = 0
            else:
                wtime = time.perf_counter()
                for it in range(0, nits):
                    sgn = 1.0
                    _cfft2(n, x, y, w, sgn, _step)
                    sgn = -1.0
                    _cfft2(n, y, x, w, sgn, _step)
                wtime = time.perf_counter() - wtime
                wtotal += wtime

                flops = 2 * nits * 5 * n * ln2
                mflops = flops / 1.0E+06 / wtime

                print(f" {round(wtime, 2):>12} {round(wtime / (2 * nits), 2):>12} {round(mflops, 2):>12}")

        if ln2 % 4 == 0:
            nits = nits // 10

        if nits < 1:
            nits = 1

    print("Elapsed time : %.6f" % wtotal)
