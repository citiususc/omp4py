import typing
import copy

__all__ = ['new_var', 'copy_var']
T = typing.TypeVar('T')


def new_var(src: T) -> T:
    return src.__class__.__new__(src.__class__)


def copy_var(org: T) -> T:
    return copy.deepcopy(org)
