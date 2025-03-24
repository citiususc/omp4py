import pytest
import os
from test.utils import proctest


def pytest_addoption(parser):
    parser.addini("timeout", "test timeout in seconds")


def pytest_collection_modifyitems(session, config, items):
    proctest_t = proctest(timeout=int(config.getini("timeout")))
    for item in items:
        item._obj = proctest_t(item._obj)
