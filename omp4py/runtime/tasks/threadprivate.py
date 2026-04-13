from __future__ import annotations

import copy

# BEGIN_CYTHON_IMPORTS
from omp4py.runtime.lowlevel.mutex import Mutex
from omp4py.runtime.lowlevel.numeric import new_pyint_array, pyint, pyint_array
from omp4py.runtime.tasks.context import omp_ctx

# END_CYTHON_IMPORTS

# BEGIN_CYTHON_IGNORE
__all__ = ["TPrivRef", "copy_private", "map_privates", "threadprivate", "threadprivates", "update_privates"]

_threadprivate_ids_len: pyint
# END_CYTHON_IGNORE

_threadprivate_ids: dict[str, pyint] = {}
_threadprivate_ids_len = 0
_threadprivate_ids_mutex = Mutex.new()


class TPrivRef:
    v: object

    @staticmethod
    def new() -> TPrivRef:
        obj: TPrivRef = TPrivRef.__new__(TPrivRef)
        obj.v = None
        return obj


def threadprivate(name: str, value: object) -> pyint:
    i = _threadprivate_ids.get(name)
    if i is None:
        global _threadprivate_ids_len
        _threadprivate_ids_mutex.lock()
        _threadprivate_ids[name] = i = _threadprivate_ids_len = len(_threadprivate_ids)
        _threadprivate_ids_mutex.unlock()
    if value is not None:
        update_privates(omp_ctx().tpvars)
        threadprivates(i).v = value

    return i


# BEGIN_CYTHON_IGNORE
def threadprivates(i: pyint) -> TPrivRef:
    tpvars = omp_ctx().tpvars
    if i >= _threadprivate_ids_len:
        update_privates(tpvars)
    return tpvars[i]


# END_CYTHON_IGNORE


def update_privates(tpvars: list[TPrivRef]) -> None:
    for _ in range(_threadprivate_ids_len):
        tpvars.append(TPrivRef.new())  # noqa: PERF401


def map_privates(names: tuple[str, ...]) -> pyint_array:
    ids = new_pyint_array(len(names))
    i: pyint
    for i in range(len(ids)):
        ids[i] = _threadprivate_ids.get(names[i], -1)
    return ids

def copy_private(obj: object) -> object:
    return copy.copy(obj) # TODO: replicate declare reduction copy strategy
