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
import platform
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


def cache_dir() -> str:
    """Return the default directory used to store omp4py cache files.

    The cache directory is resolved based on the current operating system:

    - On Windows, the value of the `TMP` environment variable is used.
    - On macOS (Darwin), `~/Library/Caches` is used.
    - On Linux, `~/.cache` is used.

    In all cases, a subdirectory named `omp4py` is appended to the
    selected base path.

    If no suitable system cache directory is found or accessible,
    a fallback directory `~/.omp4py` is returned.

    Returns:
        str: Absolute path to the directory where cache files should be stored.
    """
    parent: str | None = None
    system: str = platform.system()
    if system == "Windows":
        parent = os.getenv("TMP")
    elif system == "Darwin":
        parent = os.path.expanduser("~/Library/Caches")  # noqa: PTH111
    elif system == "Linux":
        parent = os.path.expanduser("~/.cache") # noqa: PTH111

    if parent and os.path.isdir(parent):  # noqa: PTH112
        return os.path.join(parent, "omp4py")  # noqa: PTH118

    # last fallback
    return os.path.expanduser("~/.omp4py") # noqa: PTH111


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

        cache (ClassVar[str]):
            Directory used to store cached artifacts generated during
            preprocessing and optional compilation steps.
            Controlled by the `OMP4PY_CACHE_DIR` environment variable.

        is_module (bool):
            Indicates whether the transformation is being applied to a full
            module (`True`) or to a smaller unit such as a function or class.

        filename (str): Filename of the source module.

        alias (str):
            Name of the function used in source code to express OpenMP-like
            directives (e.g., `omp(...)`).

        compile (bool):
            Enables compilation of the transformed Python code after the
            preprocessing step.

        ignore_cache (bool):
            If enabled, bypasses any existing cached artifacts and forces
            regeneration of all preprocessing and compilation outputs.
            This ensures that transformations are always performed from
            the original source, ignoring previously stored results.
            Controlled by the `OMP4PY_IGNORE_CACHE` environment variable.

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
    cache: ClassVar[str] = os.environ.get("OMP4PY_CACHE_DIR") or cache_dir()
    is_module: bool = False
    filename: str = "" # set by preprocessor
    alias: str = "omp"
    compile: bool = False
    ignore_cache: bool = environ_bool("OMP4PY_IGNORE_CACHE", default=False)
    modifiers: dict[str, bool] = field(default_factory=dict)
    device: str = "omp"
    compiler: str = "cython"
    parsers: list[str] = field(default_factory=list)
    compiler_args: dict[str, Any] = field(default_factory=dict)
    modifiers_args: dict[str, Any] = field(default_factory=dict)
    dump: str | bool = environ_bool("OMP4PY_DUMP", default=False)
    debug: bool = environ_bool("OMP4PY_DEBUG", default=False)
