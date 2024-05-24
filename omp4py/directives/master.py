import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call


@directive(name="master")
def master(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    return [ast.copy_location(ast.If(test=new_function_call("_omp_runtime.master"), body=body, orelse=[]),
                              ctx.with_node)]
