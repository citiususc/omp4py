import ast
from typing import List, Dict
from omp4py.core import _omp_clauses, directive, BlockContext, new_function_call, OmpVariableSearch
from omp4py.error import OmpSyntaxError


@directive(name="section")
def section(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    if not hasattr(ctx.with_node, "section_id"):
        raise OmpSyntaxError("section must be used inside sections", ctx.filename, ctx.with_node)

    section_assign = []
    if hasattr(ctx.with_node, "section_var"):
        section_assign = [ast.Assign(targets=[ast.Name(id=ctx.with_node.section_var, ctx=ast.Store())],
                                     value=ast.Constant(value=ctx.with_node.section_i))]

    section_call = new_function_call(ctx.with_node.section_id)
    section_call.args.append(ast.Constant(value=ctx.with_node.section_i))
    section_call.args.append(ast.Constant(value=ctx.with_node.section_n))

    # we need to handle sections variables
    OmpVariableSearch(ctx).apply()

    # clauses that affect to variables
    for clause in ["reduction", "private", "lastprivate", "firstprivate", ]:
        if clause in ctx.with_node.section_clauses:
            # used variables checked in sections directive
            _omp_clauses[clause](body, ctx.with_node.section_clauses[clause], ctx)

    return [ast.copy_location(ast.If(test=section_call, body=section_assign + body, orelse=[]), ctx.with_node)]
