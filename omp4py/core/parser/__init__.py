from __future__ import annotations

from omp4py.core.parser.tree import *  # noqa: F403
from omp4py.core.parser.error import syntax_error


# TODO: take this to create parser interface
def parse(filename: str, directive: str, lineno: int, col: int, raw_source: bool) -> Directive:
    msg = "New parser is not implemented yet"
    raise NotImplementedError(msg)
