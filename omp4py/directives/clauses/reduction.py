import ast
from typing import List
from omp4py.core import clause, BlockContext, OmpSyntaxError, args_renaming, new_function_call


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
def reduction(body: List[ast.AST], args: List[str],
              ctx: BlockContext) -> List[str]:
    assign_block = []
    reduction_block = []

    used_vars = []
    op_args = args[:]
    # reduction must be done with the lock to avoid races
    with_lock = ast.With(items=[ast.withitem(context_expr=new_function_call("_omp_runtime.level_lock"))], body=[])
    reduction_block.append(with_lock)

    # each iteration is a reduction clause
    while len(op_args) > 0:
        if op_args[0].count(':') != 1:
            raise OmpSyntaxError(f"Reduction clause must be in the format (op: var,[var,...])", ctx.filename,
                                 ctx.with_node)
        op, op_args[0] = op_args[0].split(":")  # split operator and variable from first argument
        op_value = operators.get(op.strip(), None)
        if op_value is None:
            raise OmpSyntaxError(f"{op.strip()} unknown operator", ctx.filename, ctx.with_node)

        # when find a new operation is a different reduction cluase
        for i in range(len(op_args) + 1):
            if i < len(op_args) and op_args[i] is None:
                break

        renamed_args, new_args = args_renaming(body, op_args[:i], ctx)
        assign_block.append(ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store()) for name in new_args],
                                       value=ast.Constant(value=op_value[0])))
        op_vars = set()
        for j, (old_var, new_var) in enumerate(zip(renamed_args, new_args)):
            if old_var in op_vars:
                continue
            if old_var not in ctx.with_node.local_vars:
                raise OmpSyntaxError(f"undeclared {args[j]} variable", ctx.filename, ctx.with_node)

            if op_args[j] in used_vars:
                raise OmpSyntaxError(f"Variable {args[j]} appears in more than one clause reduction",
                                     ctx.filename, ctx.with_node)
            # init variable and reduction in the with
            assign_block.append(ast.Nonlocal(names=[old_var]))
            with_lock.body.append(op_value[1](old_var, new_var))
            op_vars.add(old_var)

        used_vars.extend(op_args[:i])
        op_args = op_args[i + 1:]

    # assign block must be before the body and reduction at the end
    body[0:0] = assign_block
    body.extend(reduction_block)
    return used_vars
