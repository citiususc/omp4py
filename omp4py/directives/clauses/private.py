import ast
from typing import List
from omp4py.core import clause, args_renaming, BlockContext


@clause(name="private", min_args=1, repeatable=True)
def private(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    _, new_args = args_renaming(body, args, ctx)
    # private variables are renamed and initialised to None
    body.insert(0, ast.Assign(targets=[ast.Name(id=arg, ctx=ast.Store()) for arg in sorted(set(new_args))],
                              value=ast.Constant(value=None)))
    return args
