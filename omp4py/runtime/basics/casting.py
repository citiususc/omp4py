import typing

__all__ = ['cast']

T = typing.TypeVar("T")


def cast(t: type[T], value: typing.Any) -> T:
    if isinstance(value, t):
        return value
    if value is None:
        return None
    return t(value)
