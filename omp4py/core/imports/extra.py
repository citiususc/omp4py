"""Optional dependency management utility for omp4py.

This module provides a unified mechanism for handling optional dependencies
("extras") across the omp4py package. It allows functionality to fail lazily
with clear, actionable error messages when required extras or external
dependencies are not available.

The primary purpose of this module is to standardize ImportError handling
for optional features, such as compilation support, ensuring that users
receive consistent installation instructions.
"""

from __future__ import annotations

import typing

__all__ = ["require_extra"]


def require_extra(extra_name: str, context: str | None = None, dependency: str | None = None) -> typing.NoReturn:
    """Raise an ImportError when a required optional dependency is missing.

    This function is used to enforce optional "extras" at runtime in a
    consistent way. It is intended for lazy-loaded functionality that
    depends on additional installation features (e.g. pip extras).

    The error message includes clear installation instructions using
    pip extras syntax. An optional external dependency can also be
    specified for additional guidance.

    Args:
        extra_name (str): Name of the required extra (e.g. "compile").
        dependency (str | None): Optional external dependency required for
            the feature (e.g. "setuptools").
        context (str | None): Optional name of the feature or operation
            that triggered the error, used to improve message clarity.

    Raises:
        ImportError: Always raised to indicate the missing dependency.
    """
    where: str = (
        "This feature"
        if context is None
        else f"The '{context}' feature"
    )

    message: str = (
        f"{where} requires the '{extra_name}' extra of 'omp4py'.\n\n"
        f"Install with:\n"
        f"    pip install omp4py[{extra_name}]\n\n"
        f"Or add it to your project's dependencies (e.g. pyproject.toml)."
    )

    if dependency is not None:
        message += (
            f"\nAlternatively, ensure that '{dependency}' is installed in your environment."
        )

    raise ImportError(message)
