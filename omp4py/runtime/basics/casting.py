import typing

__all__ = ['cast']

T = typing.TypeVar("T")


def cast(t: type[T], value: typing.Any) -> T:
    if isinstance(value, t):
        return value
    if value is None:
        return None
    try:
        return t(value)
    except:
        raise TypeError(f'{type(value)} cannot be cast to {t}') from None
