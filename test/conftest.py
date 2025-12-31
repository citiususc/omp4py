from test.utils import proctest
import os


def pytest_addoption(parser):
    parser.addoption("--timeout", action="store", help="test timeout in seconds", required=True)
    parser.addoption("--pure", action="store_true", help="force omp4py pure runtime")

def pytest_configure(config):
    os.environ['OMP4PY_PURE'] = str(config.getoption("pure"))


def pytest_collection_modifyitems(session, config, items):
    if len(items) > 1:
        proctest_t = proctest(timeout=int(config.getoption("timeout")))
        for item in items:
            item._obj = proctest_t(item._obj)


def pytest_report_header(config):
    import omp4py
    return (f"omp4py compiled: {'cython' in type(omp4py.omp_get_schedule).__name__}\n"
            f"timeout: {config.getoption("timeout")}")
