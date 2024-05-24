import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_function_call, new_name
from omp4py.error import OmpSyntaxError


@directive(name="section")
def sections(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    if not hasattr(ctx.with_node, "section_id") or not hasattr(ctx.with_node, "section_i") or \
            not hasattr(ctx.with_node, "section_var") or not hasattr(ctx.with_node, "section_n"):
        raise OmpSyntaxError("section must be used inside sections", ctx.filename, ctx.with_node)

    section_assign = ast.Assign(targets=[ast.Name(id=ctx.with_node.section_var, ctx=ast.Store())],
                                value=ast.Constant(value=ctx.with_node.section_i))

    section_call = new_function_call(ctx.with_node.section_id)
    section_call.args.append(ast.Constant(value=ctx.with_node.section_i))
    section_call.args.append(ast.Constant(value=ctx.with_node.section_n))

    return [ast.copy_location(ast.If(test=section_call, body=[section_assign] + body, orelse=[]), ctx.with_node)]
