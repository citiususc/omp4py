import ast
from typing import List, Dict
from omp4py.core import directive, BlockContext, new_name, new_function_call, _omp_clauses, OmpVariableSearch
from omp4py.error import OmpSyntaxError


@directive(name="single", clauses=["private", "firstprivate", "copyprivate", "nowait"])
def single(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    single_id = new_name("_omp_single")
    single_var = new_name("_omp_single_var")
    single_call = new_function_call("_omp_runtime.single")
    single_call.args.append(ast.Constant(value=single_id))

    if_block = ast.If(test=new_function_call(single_var), body=[], orelse=[])

    single_block = ast.With(items=[ast.withitem(context_expr=single_call,
                                                optional_vars=ast.Name(id=single_var, ctx=ast.Store()))],
                            body=[if_block])

    # we need to handle variables
    used_vars = dict()
    ctx.with_node.local_vars = OmpVariableSearch(ctx).local_vars

    if "copyprivate" in clauses:
        ctx.with_node.single_var = single_var
        vars_in_clause, copy = _omp_clauses["copyprivate"](body, clauses["copyprivate"], ctx)
        if_block.orelse.extend(copy)
        used_vars.update({v: "copyprivate" for v in vars_in_clause})

    # clauses that affect to variables
    for clause in ["private", "firstprivate"]:
        if clause in clauses:
            for var in clauses[clause]:
                if var in used_vars:
                    raise OmpSyntaxError(f"Variable '{var}' cannot be used in {used_vars[var]} and {clause} "
                                         "simultaneously", ctx.filename, ctx.with_node)
            vars_in_clause = _omp_clauses[clause](body, clauses[clause], ctx)
            used_vars.update({v: clause for v in vars_in_clause})

    if "nowait" in clauses:
        _omp_clauses["nowait"](single_call, clauses["nowait"])

    if_block.body = body

    return [ast.copy_location(single_block, ctx.with_node)]
