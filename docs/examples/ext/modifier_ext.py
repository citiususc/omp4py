"""Two AST modifiers: one that runs after the transformer and one that runs before it."""

import ast

from omp4py.core.ext import Modifier, add_modifier


@add_modifier
class StripAsserts(Modifier, name="strip_asserts"):
    """Runs AFTER the transformer (the default): removes every `assert` from the generated code."""

    def visit_Assert(self, node: ast.Assert) -> None:
        return None


@add_modifier
class Xrange(Modifier, name="xrange", requires=[]):
    """Runs BEFORE the transformer (no requirements): accepts the Python 2 spelling `xrange(...)`.

    A worksharing `for` only accepts `range(...)` loops. If this modifier ran after the
    transformer it would be too late: the loop would already have been rejected.
    """

    def visit_Call(self, node: ast.Call) -> ast.Call:
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "xrange":
            node.func.id = "range"
        return node
