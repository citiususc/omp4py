import typing
from array import array
# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
from omp4py.runtime.basics.types import *

# END_CYTHON_IMPORTS


__all__ = ['array', 'iview', 'fview', 'new_int', 'new_float', 'int_from', 'float_from', 'as_iview', 'as_fview', 'copy']

iview: typing.TypeAlias = array[pyint]
fview: typing.TypeAlias = array[pyfloat]


def new_int(n: pyint) -> array[pyint]:
    return int_from([0] * n)


def new_float(n: pyint) -> array[pyfloat]:
    return float_from([0] * n)


def int_from(elems: list[pyint]) -> array[pyint]:
    return array('q', elems)


def float_from(elems: list[pyfloat]) -> array[pyfloat]:
    return array('d', elems)


def as_iview(elems: array[pyint]) -> iview:
    return elems


def as_fview(elems: array[pyfloat]) -> fview:
    return elems


def copy(elems: array) -> array:
    return elems[:]
