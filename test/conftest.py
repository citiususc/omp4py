"""Test isolation for omp4py.

A deadlock or a crash in the runtime must not stop the whole suite, so the tests run in a worker process
(``pytest-xdist`` with one worker). If a test hangs, a watchdog kills the worker, the test fails ("worker
crashed") and xdist starts a new worker for the remaining tests. The traceback of the hung threads is shown
at the end of the run.

``--no-isolate`` runs everything in the pytest process, with no worker and no watchdog (to use a debugger).
"""

import faulthandler
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import pytest

__all__ = []


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--timeout", action="store", help="test timeout in seconds", required=True)
    parser.addoption("--pure", action="store_true", help="force omp4py pure runtime")
    parser.addoption("--no-isolate", action="store_true", help="run all the tests in the pytest process (no worker)")


@pytest.hookimpl(tryfirst=True)
def pytest_cmdline_main(config: pytest.Config) -> None:
    """Run the tests in one xdist worker, unless ``--no-isolate`` or ``-n`` are given."""
    if hasattr(config, "workerinput") or config.getoption("no_isolate"):
        return
    # File where the workers write the traceback of a hung test (they inherit the environment)
    os.environ["OMP4PY_TIMEOUT_LOG"] = str(Path(tempfile.gettempdir()) / f"omp4py-timeout-{os.getpid()}.log")
    if not config.pluginmanager.hasplugin("xdist"):
        msg = "pytest-xdist is required to isolate the tests (`uv sync --group dev`), or use --no-isolate"
        raise pytest.UsageError(msg)
    if config.getoption("numprocesses", None) is None:
        config.option.numprocesses = 1
        if config.option.maxworkerrestart is None:  # by default xdist stops after 4 crashed workers
            config.option.maxworkerrestart = str(2**31 - 1)
        # pytest-cov decides how to measure before this conftest is loaded, from the command line: switch it
        # to the controller of distributed tests, which collects the data of the worker.
        cov = config.pluginmanager.get_plugin("_cov")
        if cov is not None and cov.cov_controller is not None:
            from pytest_cov import engine

            if isinstance(cov.cov_controller, engine.Central):
                cov.cov_controller.cov.stop()
                cov.start(engine.DistMaster)


@pytest.hookimpl(trylast=True)
def pytest_configure(config: pytest.Config) -> None:
    os.environ["OMP4PY_IGNORE_CACHE"] = "1"
    os.environ["OMP4PY_PURE"] = str(config.getoption("pure"))

    # With a worker, `-v` prints every test twice (when it starts and when it ends): keep only the result
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None and config.getoption("numprocesses", None):
        write = reporter.write_ensure_prefix
        reporter.write_ensure_prefix = lambda prefix, extra="", **kw: write(prefix, extra, **kw) if extra or kw else None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        run = item.obj  # ty: ignore[unresolved-attribute]

        # A new thread for every test: the thread-local state of the runtime (omp_set_num_threads...) is reset
        def in_new_thread(*args: Any, run: Any = run, **kwargs: Any) -> None:
            __tracebackhide__ = True  # keep these frames out of the traceback of a failed test
            errors: list[BaseException] = []

            def target() -> None:
                __tracebackhide__ = True
                try:
                    run(*args, **kwargs)
                except BaseException as ex:  # noqa: BLE001
                    errors.append(ex)

            thread = threading.Thread(target=target)
            thread.start()
            thread.join()
            if errors:
                raise errors[0]

        item.obj = in_new_thread  # ty: ignore[unresolved-attribute]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item: pytest.Item, nextitem: pytest.Item | None) -> Any:
    if item.config.getoption("no_isolate"):
        return (yield)

    # Watchdog: if the test hangs, the worker dies after dumping the traceback of its threads
    with Path(os.environ["OMP4PY_TIMEOUT_LOG"]).open("a") as log:
        faulthandler.dump_traceback_later(int(item.config.getoption("timeout")), exit=True, file=log)
        try:
            return (yield)
        finally:
            faulthandler.cancel_dump_traceback_later()


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    log = Path(os.environ.get("OMP4PY_TIMEOUT_LOG", ""))
    if not hasattr(config, "workerinput") and log.is_file():  # the workers also run this hook
        if text := log.read_text().strip():
            terminalreporter.section("tracebacks of the tests that timed out")
            terminalreporter.write_line(text)
        log.unlink()


def pytest_report_header(config: pytest.Config) -> str:
    import omp4py

    runtime = "native" if "cython" in type(omp4py.omp_get_schedule).__name__ else "pure"
    return f"omp4py runtime: {runtime}\ntimeout: {config.getoption('timeout')}"
