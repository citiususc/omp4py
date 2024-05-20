import ast
import inspect
from typing import List, TypeVar
from omp4py.context import OpenMPContext

_omp_context = OpenMPContext()

T = TypeVar("T")


# Inner implementation of omp function when is used as decorator
def omp_parse(f: T) -> T:
    sourcecode = inspect.getsource(f)
    filename = inspect.getsourcefile(f)
    # create a fake offset to preserve the source code line number
    source_offset = "\n" * (f.__code__.co_firstlineno - 1)
    source_ast = python_parse(source_offset + sourcecode, location=True)

    # global and local enviroment to compile the ast
    caller_frame = inspect.currentframe().f_back.f_back  # user -> api -> core

    transformer = PyOmpTransformer(source_ast, filename, caller_frame.f_globals, caller_frame.f_locals)
    omp_ast = transformer.visit(source_ast)
    # we need an import to reference pyomp functions
    omp_ast.body.insert(0, python_parse('import pyomp as _omp_pyomp').body[0])
    ast.copy_location(omp_ast.body[0], omp_ast.body[1])
    omp_ast = ast.fix_missing_locations(omp_ast)
    ompcode = compile(omp_ast, filename=filename, mode="exec")

    exec(ompcode, transformer.global_env, transformer.local_env)

    return transformer.local_env[f.__name__]


# Create ast from string with and without location
def python_parse(src: str, location=False) -> ast.Module:
    node = ast.parse(src, "<pyomp>")

    if not location:
        for child in ast.walk(node):
            if hasattr(child, "lineno"):
                delattr(child, "lineno")
            if hasattr(child, "end_lineno"):
                delattr(child, "end_lineno")

    return node


# Traverse python ast tree to find 'with omp(...):'
class PyOmpTransformer(ast.NodeTransformer):

    def __init__(self, root_node: ast.AST, filename: str, global_env: dict, local_env: dict):
        self.root_node: ast.AST = root_node
        self.filename: str = filename
        self.global_env: dict = global_env
        self.local_env: dict = local_env
        self.stack: List[ast.AST] = list()

    def visit(self, node):
        self.stack.append(node)
        new_node = super().visit(node)
        self.stack.pop()
        return new_node

    # return True if the node is an omp call in the actual enviroment
    def is_omp_function(self, node: ast.AST):
        try:
            exp = compile(ast.Expression(node), filename=self.filename, mode='eval')
            omp_candidate = eval(exp, self.global_env, self.local_env)

            from omp4py.api import omp
            if omp_candidate == omp:
                return True
        except:
            return False  # Ignore if function is not imported

    # remove @omp decorator from a class or function
    def remove_decorator(self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        node.decorator_list = [exp for exp in node.decorator_list if not self.is_omp_function(exp)]
        return self.generic_visit(node)

    # @omp decorator con be used in functions
    def visit_FunctionDef(self, node: ast.FunctionDef):
        return self.remove_decorator(node)

    # @omp decorator con be used in async functions
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        return self.remove_decorator(node)

    # @omp decorator con be used in class
    def visit_ClassDef(self, node: ast.ClassDef):
        return self.remove_decorator(node)

    # Perform OpenMP transformations if is a 'with omp(...):'
    def visit_With(self, node: ast.With):
        return node.body  # TODO
