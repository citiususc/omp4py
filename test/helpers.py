"""Helpers shared by the tests of omp4py.

A test describes the code to run as text, so the directives can be parametrized (schedule, number of threads,
iteration space...) without writing a function for every combination::

    f = omp_function('''
        def f(out):
            with omp("parallel for $schedule num_threads(3)"):
                for i in range(10):
                    out.append(i)
    ''', schedule="schedule(dynamic, 2)")
"""

import atexit
import contextlib
import hashlib
import importlib.util
import shutil
import string
import sys
import tempfile
import textwrap
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from omp4py import omp

__all__ = ["omp_function", "raises_in_the_team"]

_directory = Path(tempfile.mkdtemp(prefix="omp4py-test-"))
atexit.register(shutil.rmtree, _directory, ignore_errors=True)


def omp_function(source: str, **fields: object) -> Callable[..., Any]:
    """Create the function `f` defined in `source` and process it with omp4py.

    The text is dedented and every `$name` is replaced with `fields["name"]`. The function is processed here,
    so a `SyntaxError` of the preprocessor is raised by this call.

    Args:
        source: Code that defines a function called `f`. `from omp4py import *` is already imported.
        **fields: Values for the `$name` placeholders.

    Returns:
        The function `f`, processed by omp4py.
    """
    code = string.Template(textwrap.dedent(source)).substitute(fields)
    name = "omp_test_" + hashlib.sha1(code.encode()).hexdigest()[:16]
    path = _directory / f"{name}.py"
    if not path.exists():
        path.write_text("from omp4py import *\n\n\n" + code.strip() + "\n", encoding="utf-8")

    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return omp(module.f)


@contextlib.contextmanager
def raises_in_the_team(exception: type[BaseException]) -> Iterator[None]:
    """Check that a parallel region raises `exception` when every thread of the team raises it.

    The master thread raises it to the caller, but the other threads of the team have nobody to raise it to, and it
    ends in `threading.excepthook`. Here that hook is replaced while the region runs, so the exceptions of the other
    threads are collected instead of being reported by pytest in whichever test is running later, and the threads
    are joined (the master has already created all of them) so none of them is left running.
    """
    before = set(threading.enumerate())
    errors: list[BaseException | None] = []
    previous, threading.excepthook = threading.excepthook, lambda args: errors.append(args.exc_value)
    try:
        with pytest.raises(exception):
            yield
        for thread in set(threading.enumerate()) - before:
            thread.join()
    finally:
        threading.excepthook = previous
    assert all(isinstance(error, exception) for error in errors), errors
