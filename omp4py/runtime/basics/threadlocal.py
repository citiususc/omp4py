import typing
from threading import local, get_ident

__all__ = ['get_storage', 'has_storage', 'set_storage']

_storage: local = local()


def has_storage() -> bool:
    return hasattr(_storage, 'omp')


def get_storage() -> typing.Any:
    return getattr(_storage, 'omp')


def set_storage(value: typing.Any) -> None:
    setattr(_storage, 'omp', value)


def thread_id() -> int:
    return get_ident()
