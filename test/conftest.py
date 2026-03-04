import functools
import multiprocessing
import os
from collections.abc import Callable
from typing import Any

import pytest

__all__ = []

def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--timeout", action="store", help="test timeout in seconds", required=True)
    parser.addoption("--pure", action="store_true", help="force omp4py pure runtime")


def pytest_configure(config: pytest.Config)-> None:
    os.environ["OMP4PY_PURE"] = str(config.getoption("pure"))


def worker(q: multiprocessing.Queue, f: Callable[..., Any], timeout: float | None, *args, **kwargs)-> None:
    import coverage
    import faulthandler

    faulthandler.enable()
    if timeout is not None:
        faulthandler.dump_traceback_later(timeout, exit=True)
    if "COV_CORE_SOURCE" in os.environ:
        coverage.process_startup()
    try:
        f(*args, **kwargs)
    except BaseException as ex:
        q.put(str(ex))
        q.close()
        q.join_thread()
        raise


def isolate(timeout: float | None, f: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(f)
    def test(*args, **kwargs) -> None:
        q = multiprocessing.Queue()
        process = multiprocessing.Process(target=worker, args=[q, f, timeout, *args], kwargs=kwargs)
        process.start()
        process.join(timeout * 2 if timeout is not None else None)
        if process.is_alive():
            process.kill()
            pytest.fail("test timeout", pytrace=False)

        if process.exitcode != 0:
            if q.empty():
                pytest.fail(f"test terminated unexpectedly with code {process.exitcode}", pytrace=False)
            pytest.fail(q.get(), pytrace=False)

    return test


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    if len(items) > 0:
        timeout: str | None = config.getoption("timeout")
        for item in items:
            item.obj = isolate(timeout if timeout is None else int(timeout), item.obj)


def pytest_report_header(config: pytest.Config) -> str:
    import omp4py

    runtime = "cython" if "native" in type(omp4py.omp_get_schedule).__name__ else "pure"
    return f"omp4py runtime: {runtime}\ntimeout: {config.getoption('timeout')}"
