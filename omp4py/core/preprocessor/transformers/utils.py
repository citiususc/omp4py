import ast
from collections.abc import Callable, Generator

__all__ = ["fix_body_locations", "is_unpack_if", "unpack_if", "walk"]


def walk(node: ast.AST, descend: Callable[[ast.AST], bool] = lambda n: True) -> Generator[ast.AST]:
    """Recursively walk the AST starting from `node`, with optional control over descent.

    This function behaves similarly to `ast.walk`, but allows the caller to
    decide whether to recurse into the children of a given node.

    Args:
        node (ast.AST): The root AST node to start walking from.
        descend (Callable[[ast.AST], bool], optional): A function that takes an
            AST node and returns True if the walker should recurse into its
            children, or False to skip them. Defaults to a function that always returns True.

    Yields:
        ast.AST: Each node in the AST, in pre-order traversal (parent before children).
    """
    yield node
    if descend(node):
        for child in ast.iter_child_nodes(node):
            yield from walk(child, descend)


def fix_body_locations(body: list[ast.stmt]) -> list[ast.stmt]:
    """Apply `ast.fix_missing_locations` to each statement in a body.

    Args:
        body (list[ast.stmt]): A list of AST statement nodes representing
            a syntactic body (e.g., the body of a function, class, or module).

    Returns:
        list[ast.stmt]: A new list of statements where each node has had
            missing location information filled in recursively.
    """
    return [ast.fix_missing_locations(node) for node in body]


def unpack_if(body: list[ast.stmt]) -> ast.If:
    return ast.fix_missing_locations(ast.If(ast.Constant("_omp_unpack"), body))


def is_unpack_if(if_: ast.If) -> bool:
    match if_.test:
        case ast.Constant(value="_omp_unpack"):
            return True
    return False
