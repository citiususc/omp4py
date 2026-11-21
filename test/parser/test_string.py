import ast
import pathlib

import pytest

from omp4py.core.parser.parser import extract_directive

string_1 = "   parallel"

result_1 = \
        """    parallel """

string_2 = r"   parallel"

result_2 = \
        """     parallel """

string_3 = "parallel" "for"

result_3 = \
        """ parallel   for """

string_4 = "parallel" \
           "for"

result_4 = \
        """ parallel 
            for """

string_5 = "parallel  \
            for"

result_5 = \
        """ parallel   
            for """

def test_string(subtests: pytest.Subtests):
    source = pathlib.Path(__file__).read_text()
    module = ast.parse(source)
    tests : dict[str, ast.Constant] = {}
    expect : dict[str, str] = {}

    for node in module.body:
        match node:
            case ast.Assign(targets=[ast.Name(name)], value=ast.Constant() as value):
                name_parts = name.split("_", 1)
                if name_parts[0] == "string":
                    tests[name_parts[1]] = value
                elif name_parts[0] == "result":
                    expect[name_parts[1]] = str(value.value)

    assert tests.keys() == expect.keys()
    for e in tests:
        with subtests.test(msg=f"{e}"):
            result = extract_directive(tests[e], source,__file__)
            assert result == expect[e]
