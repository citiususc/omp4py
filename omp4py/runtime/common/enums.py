from __future__ import annotations
import typing
# BEGIN_CYTHON_IMPORTS: Add 'cython.imports' prefix to omp4py packages
from omp4py.runtime.basics.types import *

# END_CYTHON_IMPORTS


__all__ = ['omp_sched_t',
           'omp_sched_static',
           'omp_sched_dynamic',
           'omp_sched_guided',
           'omp_sched_auto',
           'omp_sched_monotonic',
           'omp_enum_names']


class BaseEnum:
    _value: pyint

    def __init__(self, value: pyint):
        self._value = value

    def __and__(self, other: BaseEnum) -> typing.Self:
        return self.__class__(self._value & other._value)

    def __or__(self, other: BaseEnum) -> typing.Self:
        return self.__class__(self._value | other._value)

    def __xor__(self, other: BaseEnum) -> typing.Self:
        return self.__class__(self._value ^ other._value)

    def __invert__(self) -> typing.Self:
        return self.__class__(~self._value)

    def __lshift__(self, n: pyint) -> typing.Self:
        return self.__class__(self._value << n)

    def __rshift__(self, n: pyint) -> typing.Self:
        return self.__class__(self._value >> n)

    def __eq__(self, other: BaseEnum):
        if not isinstance(other, self.__class__):
            return False
        return self._value == other._value

    def __ne__(self, other: BaseEnum):
        if not isinstance(other, self.__class__):
            return True
        return self._value != other._value

    def __int__(self) -> pyint:
        return self._value


class omp_sched_t(BaseEnum):
    def __init__(self, value: pyint):
        super().__init__(value)


omp_sched_static: omp_sched_t = omp_sched_t(0x1)
omp_sched_dynamic: omp_sched_t = omp_sched_t(0x2)
omp_sched_guided: omp_sched_t = omp_sched_t(0x3)
omp_sched_auto: omp_sched_t = omp_sched_t(0x4)
omp_sched_monotonic: omp_sched_t = omp_sched_t(0x80000000)

globals()['omp_sched_static'] = omp_sched_static
globals()['omp_sched_dynamic'] = omp_sched_dynamic
globals()['omp_sched_guided'] = omp_sched_guided
globals()['omp_sched_auto'] = omp_sched_auto
globals()['omp_sched_monotonic'] = omp_sched_monotonic

omp_enum_names: dict[str, BaseEnum] = {name: value for name, value in globals().items() if isinstance(value, BaseEnum)}
globals()['omp_enum_names'] = omp_enum_names
