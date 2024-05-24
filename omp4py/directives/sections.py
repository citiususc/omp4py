import ast
from typing import List, Dict
from omp4py.core import directive, _omp_clauses, BlockContext, OmpVariableSearch, new_function_call, new_name
from omp4py.error import OmpSyntaxError


@directive(name="sections", clauses=["private", "firstprivate", "lastprivate", "reduction", "nowait"])
def sections(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    new_body = list()
    sections_id = new_name("_omp_sections")
    section_var = new_name("_omp_section_var")
    sections_call = new_function_call("_omp_runtime.open_sections")
    sections_call.args.append(ast.Constant(value=sections_id))

    with_block = ast.With(items=[ast.withitem(context_expr=sections_call,
                                              optional_vars=ast.Name(id=sections_id, ctx=ast.Store()))])

    if "lastprivate" in clauses:
        new_body.append(ast.Assign(targets=[ast.Name(id=section_var, ctx=ast.Store())], value=ast.Constant(value=None)))
        ctx.with_node.lastprivate = ast.Name(id=section_var, ctx=ast.Load())

    # check that inner blocks are section blocks and set the sections id for each
    for i, elem in enumerate(body):
        if isinstance(elem, ast.With) and len(elem.items) == 1:
            elem_exp = elem.items[0].context_expr
            if isinstance(elem_exp, ast.Call) and len(elem_exp.args) == 1:
                elem_arg = elem_exp.args[0]
                if isinstance(elem_arg, ast.Constant) and elem_arg.value.strip() == "section":
                    elem.section_id = sections_id
                    elem.section_i = i
                    elem.section_var = section_var
                    elem.section_n = len(body)
                    continue
        raise OmpSyntaxError("sections can only contains one or more section", ctx.filename, ctx.with_node)

    # we need to handle variables
    used_vars = dict()
    if not hasattr(ctx.with_node, "local_vars"):
        ctx.with_node.local_vars = OmpVariableSearch(ctx).local_vars

    # clauses that affect to variables
    for clause in ["private", "lastprivate", "firstprivate", "reduction"]:
        if clause in clauses:
            for var in clauses[clause]:
                if var in used_vars:
                    raise OmpSyntaxError(f"Variable '{var}' cannot be used in {used_vars[var]} and {clause} "
                                         "simultaneously", ctx.filename, ctx.with_node)
            vars_in_clause = _omp_clauses[clause](body, clauses[clause], ctx)
            used_vars.update({v: clause for v in vars_in_clause})

    if "nowait" in clauses:
        _omp_clauses["nowait"](sections_call, clauses["nowait"])

    with_block.body = body
    ast.copy_location(with_block, ctx.with_node)
    new_body.append(with_block)
    return new_body
