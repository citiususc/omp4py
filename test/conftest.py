from test.utils import proctest
from omp4py import omp4py_compiled

def pytest_addoption(parser):
    parser.addini("timeout", "test timeout in seconds")


def pytest_collection_modifyitems(session, config, items):
    if len(items) > 1:
        proctest_t = proctest(timeout=int(config.getini("timeout")))
        for item in items:
            item._obj = proctest_t(item._obj)

def pytest_report_header(config):
    return f"omp4py compiled: {omp4py_compiled}"