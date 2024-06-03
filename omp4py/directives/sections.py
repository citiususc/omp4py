import ast
from typing import List, Dict
from omp4py.core import directive, _omp_clauses, BlockContext, OmpVariableSearch, new_function_call, new_name, \
    is_omp_function
from omp4py.error import OmpSyntaxError


@directive(name="sections", clauses=["private", "firstprivate", "lastprivate", "reduction", "nowait"])
def sections(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    new_body = list()
    sections_id = new_name("_omp_sections")

    sections_call = new_function_call("_omp_runtime.sections")
    sections_call.args.append(ast.Constant(value=sections_id))

    with_block = ast.With(items=[ast.withitem(context_expr=sections_call,
                                              optional_vars=ast.Name(id=sections_id, ctx=ast.Store()))])

    # variable to store the last section executed
    if "lastprivate" in clauses:
        section_var = new_name("_omp_section_var")
        new_body.append(ast.copy_location(
            ast.Assign(targets=[ast.Name(id=section_var, ctx=ast.Store())], value=ast.Constant(value=None)),
            ctx.with_node))
        ctx.with_node.lastprivate = ast.copy_location(ast.Name(id=section_var, ctx=ast.Load()), ctx.with_node)
    else:
        ctx.with_node.lastprivate = None
        section_var = None

    # variables must be handled by each section
    var_clauses = {n: clauses[n] for n in clauses if n in ["reduction", "private", "lastprivate", "firstprivate"]}

    # we need to handle variables
    if hasattr(ctx.with_node, "used_vars"):
        used_vars = ctx.with_node.used_vars
    else:
        used_vars = dict()

    # clauses that affect to variables
    OmpVariableSearch(ctx).apply()
    local_vars = ctx.with_node.local_vars.copy()  # preserve variables
    body_bak = ctx.with_node.body
    ctx.with_node.body = []
    for clause in ["reduction", "private", "lastprivate", "firstprivate", ]:
        if clause in clauses:
            vars_in_clause = _omp_clauses[clause]([], clauses[clause], ctx)
            for var in vars_in_clause:
                if var in used_vars:
                    raise OmpSyntaxError(f"variable '{var}' cannot be used in {used_vars[var]} and {clause} "
                                         "simultaneously", ctx.filename, ctx.with_node)
            used_vars.update({v: clause for v in vars_in_clause})
    ctx.with_node.body = body_bak
    ctx.with_node.local_vars = local_vars

    # check that inner blocks are section blocks and set the sections id for each
    for i, elem in enumerate(body):
        if isinstance(elem, ast.With) and len(elem.items) == 1:
            elem_exp = elem.items[0].context_expr
            if (isinstance(elem_exp, ast.Call) and len(elem_exp.args) == 1 and
                    is_omp_function(elem_exp.func, ctx.global_env, ctx.local_env) and
                    isinstance(elem_exp.args[0], ast.Constant) and elem_exp.args[0].value.strip() == "section"):
                elem.section_id = sections_id
                elem.section_i = i
                elem.section_n = len(body)
                elem.section_clauses = var_clauses
                if section_var is not None:
                    elem.lastprivate = ctx.with_node.lastprivate
                    elem.section_var = section_var
                continue
        raise OmpSyntaxError("sections can only contains one or more section", ctx.filename, ctx.with_node)

    if "nowait" in clauses:
        _omp_clauses["nowait"](sections_call, clauses["nowait"], ctx)

    with_block.body = body
    ast.copy_location(with_block, ctx.with_node)
    new_body.append(with_block)
    return new_body
