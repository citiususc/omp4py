import ast
from typing import List, Dict, Set
from omp4py.core import directive, BlockContext
from omp4py.error import OmpSyntaxError


class OmpThreadprivate(ast.NodeTransformer):

    def __init__(self, thread_vars: Set[str]):
        self.thread_vars: Set[str] = thread_vars

    def visit_Name(self, node: ast.Name):
        if node.id in self.thread_vars:
            return ast.copy_location(
                ast.Attribute(value=ast.copy_location(
                    ast.Attribute(value=ast.copy_location(ast.Name(id='_omp_runtime', ctx=ast.Load()), node),
                                  attr="var", ctx=ast.Load()), node), attr=node.id, ctx=node.ctx), node)
        return node


@directive(name="threadprivate", max_args=-1)
def threadprivate(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    new_body = list()
    if not (len(body) == 1 and isinstance(body[0], ast.Pass)):
        raise OmpSyntaxError("threadprivate directive only allows pass as body", ctx.filename, ctx.with_node)

    thread_vars = set(clauses["threadprivate"])
    OmpThreadprivate(thread_vars).visit(ctx.root_node)

    return new_body
