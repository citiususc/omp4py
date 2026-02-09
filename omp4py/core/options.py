from __future__ import annotations

import os
from typing import ClassVar
from dataclasses import dataclass

__all__ = ["Options"]


def environ_bool(key: str, default: bool) -> bool:
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Options:
    _new_core: ClassVar[bool] = environ_bool("_OMP4PY_NEW_CORE", default=False)
    pure: ClassVar[bool] = environ_bool("OMP4PY_PURE", default=False)
    alias: str = "omp"
    debug: bool = environ_bool("OMP4PY_DEBUG", default=False)


