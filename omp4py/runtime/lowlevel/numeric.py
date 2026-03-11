"""Basic numeric type aliases used by the `omp4py` runtime.

This module defines simple aliases for Python numeric types used across
the `omp4py` runtime. These aliases help keep type annotations consistent
between the pure Python implementation and the compiled version generated
with Cython.

When `omp4py` is compiled, runtime types use different low-level
representations. For example, the Python `int` type corresponds to
`cython.longlong` in the compiled runtime. By using aliases such as
`pyint`, the source code remains the same for both the pure Python and
compiled implementations.

The module also defines simple array aliases and helper constructors for
creating numeric arrays used by the runtime.
"""

__all__ = ["new_pyfloat_array", "new_pyint_array", "pycomplex", "pyfloat", "pyint", "pyint_array"]

# Integer
type pyint = int  # noqa: PYI042
type pyint_array = list[pyint]  # noqa: PYI042


# Floating point
type pyfloat = float  # noqa: PYI042
type pyfloat_array = list[pyfloat]  # noqa: PYI042


# Complex
type pycomplex = complex  # noqa: PYI042


def new_pyfloat_array(n: pyint) -> pyfloat_array:
    """Return a zero-initialized float array of length `n`.

    Args:
        n (pyint): Array length.

    Returns:
        pyfloat_array: Array filled with `0.0`.
    """
    return [0.0] * n


def new_pyint_array(n: pyint) -> pyint_array:
    """Return a zero-initialized integer array of length `n`.

    Args:
        n (pyint): Array length.

    Returns:
        pyint_array: Array filled with `0`.
    """
    return [0] * n
