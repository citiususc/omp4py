"""Compile-time helper functions used by the `omp4py` runtime.

This module provides Python equivalents of helper operations used when
the runtime is compiled with Cython. They allow the same source code to
run correctly in Python mode while preserving valid type
information for static type checkers.

These functions act as placeholders during normal Python execution. In
compiled mode, they are handled by the compiler and do not exist as
runtime Python calls.

This makes it possible to share a single source implementation between
the interpreted and compiled versions of the runtime.
"""


def cy_typeof[T](obj: T) -> type[T]:
    """Return the runtime type of an object.

    This is the pure Python equivalent of a compile-time type query used
    in compiled mode.

    Args:
        obj (T): Object whose type is requested.

    Returns:
        type[T]: Type of the input object.
    """
    return type(obj)


def cy_cast[T](cls: type[T], obj: object) -> T:
    """Return an object cast to the requested type.

    In pure Python mode this function performs no runtime conversion and
    simply returns the original object. Its main purpose is to preserve
    typing information and mirror cast operations used in compiled mode.

    Args:
        cls (type[T]): Target type.
        obj (object): Object to cast.

    Returns:
        T: Input object viewed as the requested type.
    """
    return obj  # ty:ignore[invalid-return-type]
