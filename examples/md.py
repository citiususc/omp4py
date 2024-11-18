import numpy as np
import math
import time
import random
from omputils import njit, pyomp, omp

_PI_2 = math.pi / 2

@njit
def _v(x):
    if x < _PI_2:
        return math.sin(x) ** 2
    else:
        return 1.0

@njit
def _dv(x):
    if x < _PI_2:
        return 2 * math.sin(x) * math.cos(x)
    else:
        return 0.0


def _initialize(np, nd, box, pos, vel, acc, seed):
    r = random.Random(seed)
    for i in range(np):
        for j in range(nd):
            x = r.randint(0, 10000) / 10000
            pos[i][j] = box[j] * x
            vel[i][j] = 0.0
            acc[i][j] = 0.0

@njit
def _dist(nd, r1, r2, dr):
    d = 0.0
    for i in range(nd):
        dr[i] = r1[i] - r2[i]
        d += dr[i] * dr[i]
    return math.sqrt(d)

@njit
def _dot_prod(n, x, y):
    t = 0.0

    for i in range(n):
        t += x[i] * y[i]

    return t


@njit
def _pyomp_compute(n, nd, pos, vel, mass, f):
    pot = 0.0
    kin = 0.0

    with pyomp("parallel reduction(+:pot, kin)"):
        rij = np.empty(nd)
        with omp("for"):
            for i in range(n):
                for j in range(nd):
                    f[i][j] = 0.0
                for j in range(n):
                    if i != j:
                        d = _dist(nd, pos[i], pos[j], rij)

                        pot = pot + 0.5 * _v(d)
                        for k in range(nd):
                            if d == 0:
                                d = 10e-16
                            f[i][k] = f[i][k] - rij[k] * _dv(d) / d
                kin = kin + _dot_prod(nd, vel[i], vel[j])
    kin = kin * 0.5 * mass
    return pot, kin


@njit
def _pyomp_update(n, nd, pos, vel, f, a, mass, dt):
    rmass = 1.0 / mass
    with pyomp("parallel for"):
        for i in range(n):
            for j in range(nd):
                pos[i][j] = pos[i][j] * vel[i][j] * dt + 0.5 * dt * dt * a[i][j]
                vel[i][j] = vel[i][j] + 0.5 * dt * (f[i][j] * rmass + a[i][j])
                a[i][j] = f[i][j] * rmass


@omp
def _omp4py_compute(n, nd, pos, vel, mass, f):
    pot = 0.0
    kin = 0.0

    with omp("parallel reduction(+:pot, kin)"):
        rij = np.empty(nd)
        with omp("for"):
            for i in range(n):
                for j in range(nd):
                    f[i][j] = 0.0
                for j in range(n):
                    if i != j:
                        d = _dist(nd, pos[i], pos[j], rij)

                        pot = pot + 0.5 * _v(d)
                        for k in range(nd):
                            if d == 0:
                                d = 10e-16
                            f[i][k] = f[i][k] - rij[k] * _dv(d) / d
                kin = kin + _dot_prod(nd, vel[i], vel[j])
    kin = kin * 0.5 * mass
    return pot, kin


@omp
def _omp4py_update(n, nd, pos, vel, f, a, mass, dt):
    rmass = 1.0 / mass
    with omp("parallel for"):
        for i in range(n):
            for j in range(nd):
                pos[i][j] = pos[i][j] * vel[i][j] * dt + 0.5 * dt * dt * a[i][j]
                vel[i][j] = vel[i][j] + 0.5 * dt * (f[i][j] * rmass + a[i][j])
                a[i][j] = f[i][j] * rmass


def md(n=2000, steps=10, seed=0, numba=False):
    print(f"md: n={n}, steps={steps}, seed={seed}, numba={numba}")
    mass = 1.0
    dt = 1.0e-4
    ndim = 3
    _compute = _pyomp_compute if numba else _omp4py_compute
    _update = _pyomp_update if numba else _omp4py_update

    box = np.empty(ndim)
    position = np.empty((n, ndim))
    velocity = np.empty((n, ndim))
    force = np.empty((n, ndim))
    accel = np.empty((n, ndim))

    for i in range(ndim):
        box[i] = 10.0

    _initialize(n, ndim, box, position, velocity, accel, seed)

    wtime = time.perf_counter()
    potencial, kinetic = _compute(n, ndim, position, velocity, mass, force)
    E0 = potencial + kinetic

    for i in range(steps):
        potencial, kinetic = _compute(n, ndim, position, velocity, mass, force)
        _update(n, ndim, position, velocity, force, accel, mass, dt)
    wtime = time.perf_counter() - wtime

    print("Elapsed time : %.6f" % wtime)
