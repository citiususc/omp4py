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
from dataclasses import dataclass, field
from typing import Any, ClassVar

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
    """Configuration options for the preprocessing pipeline.

    This class defines the set of options that control how the `omp4py`
    core transforms Python code and how the resulting code is
    handled.

    Attributes:
        pure (ClassVar[bool]):
            Forces the use of the pure Python runtime, even when compiled
            extensions are available. When enabled, native modules are
            ignored in favor of their Python equivalents.
            Controlled by the `OMP4PY_PURE` environment variable.

        is_module (bool):
            Indicates whether the transformation is being applied to a full
            module (`True`) or to a smaller unit such as a function or class.

        alias (str):
            Name of the function used in source code to express OpenMP-like
            directives (e.g., `omp(...)`).

        compile (bool):
            Enables compilation of the transformed Python code after the
            preprocessing step.

        modifiers (dict[str, bool]):
            Enables or disables specific source modifiers. This allows
            applying transformations to the code before or after the main
            preprocessing step, enabling adaptation or optimization of the AST.

        device (str):
            Default target device identifier used during transformation.

        compiler (str):
            Compiler backend used when `compile` is enabled (e.g., `"cython"`).

        parsers (list[str]):
            Alternative parsers that extend directive syntax beyond the scope
            of the default `alias` parser.

        compiler_args (dict[str, Any]):
            Arguments passed to the selected compiler backend.

        modifiers_args (dict[str, Any]):
            Additional configuration values associated with directive
            modifiers.

        dump (str | bool):
            If set, writes the transformed source code to a file. If a string
            is provided, it is used as the output path. If `True`, a default
            location may be used.

        debug (bool):
            Enables debug mode, which may activate additional checks or
            logging during preprocessing.
            Controlled by the `OMP4PY_DEBUG` environment variable.
    """
    pure: ClassVar[bool] = environ_bool("OMP4PY_PURE", default=False)
    is_module: bool = False
    alias: str = "omp"
    compile: bool = False
    modifiers: dict[str, bool] = field(default_factory=dict)
    device: str = "omp"
    compiler: str = "cython"
    parsers: list[str] = field(default_factory=list)
    compiler_args: dict[str, Any] = field(default_factory=dict)
    modifiers_args: dict[str, Any] = field(default_factory=dict)
    dump: str | bool = environ_bool("OMP4PY_DUMP", default=False)
    debug: bool = environ_bool("OMP4PY_DEBUG", default=False)

