import ast
from typing import List
from omp4py.core import clause, BlockContext
from omp4py.error import OmpSyntaxError


# make sure that all loops have all three arguments
def unpack(args: List[ast.AST]) -> List[ast.AST]:
    if len(args) == 3:
        return args
    if len(args) == 2:
        return args + [ast.Constant(value=0)]
    if len(args) == 1:
        return [ast.Constant(value=0)] + args + [ast.Constant(value=1)]
    return [ast.Constant(value=None)] * 3


# Check that collapse loops meet the requirements
def recursive_check(body: List[ast.AST], deep: int, for_stack: List[ast.For], ctx: BlockContext):
    # Same requirements that a for directive
    if len(body) > 1 or not isinstance(body[0], ast.For):
        raise OmpSyntaxError("collapse clause requires that all for loops to be nested without additional statements",
                               ctx.filename, ctx.with_node)

    for_stm: ast.For = body[0]
    for_stack.append(for_stm)
    # Same requirements that a for directive again
    if not (isinstance(for_stm.iter, ast.Call) and isinstance(for_stm.iter.func, ast.Name) and
            for_stm.iter.func.id == "range" and isinstance(for_stm.target, ast.Name)):
        raise OmpSyntaxError("collapse clause requires that affected inner for loops to be range loops")

    # unpack argument and repeat for affected inner loops
    if deep > 1:
        return [unpack(for_stm.iter.args)] + recursive_check(for_stm.body, deep - 1, for_stack, ctx)
    else:
        return [unpack(for_stm.iter.args)]


@clause(name="collapse", min_args=1, max_args=1)
def collapse(body: List[ast.AST], args: List[str], ctx: BlockContext) -> ast.For:
    try:
        deep = int(args[0])
    except:
        raise OmpSyntaxError("collapse clause expects a integer argument", ctx.filename, ctx.with_node)

    if deep < 1:
        raise OmpSyntaxError("collapse expects a positive integer argument", ctx.filename, ctx.with_node)

    for_stm: ast.For = body[0]
    # collapse(1) is a simple for directive
    if deep == 1:
        return for_stm

    for_stack: List[ast.For] = list()
    new_iter_args = list()
    nested_iter = recursive_check(body, deep, for_stack, ctx)

    # stack together as tuple, start, stop y step for nested loops
    for i in range(3):
        new_iter_args.append(ast.Tuple(elts=[iter[i] for iter in nested_iter], ctx=ast.Load()))
    for_stm.iter.args = new_iter_args
    # merge loops targe variables
    for_stm.target = ast.Tuple(elts=[loop.target for loop in for_stack], ctx=ast.Store())
    # take the body of the deeper affected loop
    for_stm.body = for_stack[-1].body

    return for_stm
