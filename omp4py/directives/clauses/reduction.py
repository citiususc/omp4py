import ast
from typing import List
from omp4py.core import clause, BlockContext, OmpSyntaxError, new_name, new_function_call, var_renaming


def basic_op(op):
    return lambda target, value: ast.AugAssign(target=ast.Name(id=target, ctx=ast.Store()),
                                               op=op(),
                                               value=ast.Name(id=value, ctx=ast.Load()))


def bool_op(op):
    return lambda target, value: ast.Assign(targets=[ast.Name(id=target, ctx=ast.Store())],
                                            value=ast.BoolOp(op=op(), values=[ast.Name(id=target, ctx=ast.Load()),
                                                                              ast.Name(id=value, ctx=ast.Load())]))


# Supported reduction operators and innit values
operators = {
    "+": (0, basic_op(ast.Add)),
    "*": (1, basic_op(ast.Mult)),
    "-": (0, basic_op(ast.Sub)),
    "&": (~0, basic_op(ast.BitAnd)),
    "|": (0, basic_op(ast.BitOr)),
    "^": (0, basic_op(ast.BitXor)),
    "&&": (True, bool_op(ast.And)),
    "||": (False, bool_op(ast.Or)),
}


@clause(name="reduction", min_args=1, repeatable=None)
def reduction(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    required_nonlocal = hasattr(ctx.with_node, "new_team")
    old_local_vars = ctx.with_node.local_vars.copy()
    assign_block = []

    used_vars = set()
    op_args = args[:]
    # reduction must be done with the lock to avoid races
    with_lock = ast.With(items=[ast.withitem(context_expr=new_function_call("_omp_runtime.level_lock"))], body=[])

    # each iteration is a reduction clause
    while len(op_args) > 0:
        if op_args[0].count(':') != 1:
            raise OmpSyntaxError(f"reduction clause must be in the format (op: var,[var,...])", ctx.filename,
                                 ctx.with_node)
        op, op_args[0] = op_args[0].split(":")  # split operator and variable from first argument
        op_value = operators.get(op.strip(), None)
        if op_value is None:
            raise OmpSyntaxError(f"{op.strip()} unknown operator", ctx.filename, ctx.with_node)

        # when find a new operation is a different reduction cluase
        for i in range(len(op_args) + 1):
            if i < len(op_args) and op_args[i] is None:
                break

        for arg in set(op_args[:i]):
            if arg not in ctx.with_node.local_vars:
                raise OmpSyntaxError(f"undeclared {arg} variable", ctx.filename, ctx.with_node)

            if arg in ctx.with_node.private_vars:
                raise OmpSyntaxError(
                    "variable that appears in a reduction clause must be shared in the current parallel",
                    ctx.filename, ctx.with_node)

            if arg in used_vars:
                raise OmpSyntaxError(f"variable {arg} appears in more than one clause reduction",
                                     ctx.filename, ctx.with_node)

            ctx.with_node.local_vars[arg] = "_omp_" + new_name(arg)
            assign_block.append(ast.copy_location(
                ast.Assign(targets=[ast.Name(id=ctx.with_node.local_vars[arg], ctx=ast.Store())],
                           value=ast.Constant(value=op_value[0])), ctx.with_node))
            # init variable and reduction in the with
            if required_nonlocal:
                assign_block.append(ast.Nonlocal(names=[old_local_vars[arg]]))

            with_lock.body.append(op_value[1](old_local_vars[arg], ctx.with_node.local_vars[arg]))

        used_vars.update(op_args[:i])
        op_args = op_args[i + 1:]

    var_renaming(ctx.with_node, old_local_vars)

    # assign block must be before the body and reduction at the end
    body[0:0] = assign_block
    body.append(ast.copy_location(with_lock, ctx.with_node))
    return used_vars
