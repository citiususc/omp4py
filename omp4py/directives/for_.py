import ast
from typing import List, Dict
from omp4py.core import directive, _omp_clauses, new_name, BlockContext, OmpVariableSearch
from omp4py.error import OmpSyntaxError


# search for breaks inside omp for to raise error
class OmpBreakSearch(ast.NodeVisitor):

    def __init__(self, ctx: BlockContext, for_: ast.For):
        self.ctx: BlockContext = ctx
        for node in for_.body:
            self.visit(node)

    def visit_For(self, node: ast.For):
        return node

    def visit_While(self, node: ast.While):
        return node

    def visit_Break(self, node: ast.Break):
        raise OmpSyntaxError("for directive block cannot contain break statements", self.ctx.filename, node)


@directive(name="for",
           clauses=["private", "firstprivate", "lastprivate", "reduction", "schedule", "collapse", "ordered", "nowait"])
def for_(body: List[ast.AST], clauses: Dict[str, List[str]], ctx: BlockContext) -> List[ast.AST]:
    if len(body) > 1 or not isinstance(body[0], ast.For):
        raise OmpSyntaxError("for directive can only enclose a loop", ctx.filename, ctx.with_node)

    for_stm: ast.For = body[0]

    # omp for only work with range loops
    if not (isinstance(for_stm.iter, ast.Call) and isinstance(for_stm.iter.func, ast.Name) and
            for_stm.iter.func.id == "range" and isinstance(for_stm.target, ast.Name)):
        raise OmpSyntaxError("for directive requires a range loop", ctx.filename, ctx.with_node)

    for_id = new_name("_omp_for")
    for_stm.iter.keywords.append(ast.keyword(arg="id", value=ast.Constant(value=for_id)))
    for_stm.parallel = True  #

    if "collapse" in clauses:
        _omp_clauses["collapse"](body, clauses["collapse"], ctx)

    if "schedule" in clauses:
        for_stm.iter = _omp_clauses["schedule"](for_stm.iter, clauses["schedule"], ctx)
    else:
        for_stm.iter = _omp_clauses["schedule"](for_stm.iter, ["static"], ctx)

    if "ordered" in clauses:
        for_stm.iter = _omp_clauses["ordered"](for_stm.iter, clauses["ordered"], ctx)

    if "nowait" in clauses:
        for_stm.iter = _omp_clauses["nowait"](for_stm.iter, clauses["nowait"], ctx)

    # we need to handle variables
    used_vars = dict()
    OmpVariableSearch(ctx).apply()

    # enable lastprivate check at runtime and set the target value
    if "lastprivate" in clauses:
        for_stm.iter.keywords.append(ast.keyword(arg="lastprivate", value=ast.Constant(value=True)))
        # set the variable to use as argument in runtime lastprivate function
        if isinstance(for_stm.target, ast.Tuple):
            ctx.with_node.lastprivate = ast.Tuple(
                elts=[ast.Name(id=arg.id, ctx=ast.Load()) for arg in for_stm.target.elts], ctx=ast.Load())
            init = ast.Tuple(elts=[ast.Constant(value=None) for _ in for_stm.target.elts], ctx=ast.Load())
        else:
            init = ast.Constant(value=None)
            ctx.with_node.lastprivate = ast.Name(id=for_stm.target.id, ctx=ast.Load())
        # set an init value if the loop has no iterations.
        body.insert(0, ast.copy_location(ast.Assign(targets=[for_stm.target], value=init), ctx.with_node))

    # clauses that affect to variables
    for clause in ["reduction", "private", "lastprivate", "firstprivate"]:
        if clause in clauses:
            vars_in_clause = _omp_clauses[clause](body, clauses[clause], ctx)
            for var in vars_in_clause:
                if var in used_vars:
                    raise OmpSyntaxError(f"variable '{var}' cannot be used in {used_vars[var]} and {clause} "
                                         "simultaneously", ctx.filename, ctx.with_node)
            ctx.with_node.private_vars.update(vars_in_clause)
            used_vars.update({v: clause for v in vars_in_clause})

    OmpBreakSearch(ctx, for_stm)

    return body
