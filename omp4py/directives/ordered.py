import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_name, new_function_call
from omp4py.error import OmpSyntaxError


@directive(name="ordered", max_args=-1)
def ordered(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    for parent in ctx.stack[::-1]:
        if hasattr(parent, "ordered"):
            if hasattr(parent, "busy_ordered"):
                raise OmpSyntaxError("only a single ordered clause can appear on a loop directive",
                                     ctx.filename, ctx.with_node)
            else:
                parent.busy_ordered = True
            break
    f = new_function_call("_omp_runtime.ordered")
    f.args.append(ast.Constant(value=new_name("ordered")))
    with_block = ast.With(items=[ast.withitem(context_expr=f)], body=body)
    ast.copy_location(with_block, ctx.with_node)
    return [with_block]
