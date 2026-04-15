"""Options module for the preprocessing system.

This module defines the global configuration options used by the
preprocessor and transformer pipeline.

The options control the behavior of the compilation process, including
debugging features, runtime mode selection, and directive aliasing.

Configuration values can be defined either directly in code or via
environment variables, allowing flexible integration in different
execution environments.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import ClassVar

__all__ = ["Options"]


def environ_bool(key: str, default: bool) -> bool:
    """Read a boolean configuration value from environment variables.

    This helper interprets common truthy values such as:
    - "1", "true", "yes", "on"

    Args:
        key (str): Environment variable name.
        default (bool): Default value if the variable is not set.

    Returns:
        bool: Parsed boolean value.
    """
    return os.environ.get(key, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Options:
    """Options container.

    Attributes:
        pure (ClassVar[bool]):
            This option forces the system to use pure Python mode, even if the
            library has been compiled with native extensions, effectively
            ignoring them in favor of pure Python implementations.
            Controlled by `OMP4PY_PURE`.

        alias (str):
            Function name used in source code to define OpenMP-like directives
            (default: "omp(...)").

        debug (bool):
            Enables debug mode with additional logging or checks.
            Controlled by `OMP4PY_DEBUG`.

        dump (str | None):
            Optional path to dump the transformed AST as source code
            for debugging or inspection purposes.
    """
    pure: ClassVar[bool] = environ_bool("OMP4PY_PURE", default=False)
    alias: str = "omp"
    debug: bool = environ_bool("OMP4PY_DEBUG", default=False)
    dump: str | None = None


