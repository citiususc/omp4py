import ast
from typing import List
from omp4py.core import clause, args_renaming, new_function_call, BlockContext
from omp4py.error import OmpSyntaxError


@clause(name="firstprivate", min_args=1, repeatable=True)
def firstprivate(body: List[ast.AST], args: List[str], ctx: BlockContext) -> List[str]:
    renamed_args, new_args = args_renaming(body, [arg for arg in args], ctx)

    # variable must exists
    for i, arg in enumerate(renamed_args):
        if arg not in ctx.with_node.local_vars:
            raise OmpSyntaxError(f"undeclared {arg[i]} variable", ctx.filename, ctx.with_node)

    # firstprivate variables are renamed and initialised to previous value using a shadow copy function
    vars_assign = []
    used = set()
    for old_arg, new_arg in zip(renamed_args, new_args):
        if old_arg not in used:
            arg_copy = new_function_call("_omp_runtime.var_copy")
            arg_copy.args.append(ast.Name(id=old_arg, ctx=ast.Load()))
            vars_assign.append(ast.Assign(targets=[ast.Name(id=new_arg, ctx=ast.Store())],
                                          value=arg_copy))
        used.add(old_arg)
    body[0:0] = vars_assign
    return args
