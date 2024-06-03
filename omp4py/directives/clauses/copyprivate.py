import ast
from typing import List
from omp4py.core import clause, BlockContext, new_name, new_function_def, new_function_call


@clause(name="copyprivate", min_args=1, repeatable=True)
def copyprivate(body: List[ast.AST], args: List[str], ctx: BlockContext) -> (List[str], [ast.AST]):
    single_var = ctx.with_node.single_var

    # create a function that links variables in a thread to be called after the block ends
    update_id = new_name("_omp_copyprivate")
    update_function = new_function_def(update_id)

    update_function.body.append(ast.Nonlocal(names=[ctx.with_node.local_vars[arg] for arg in args]))
    update_function.args.vararg = ast.arg(arg='args')
    for i,arg in enumerate(args):
        update_function.body.append(ast.Assign(targets=[ast.Name(id=ctx.with_node.local_vars[arg], ctx=ast.Store())],
                                               value=ast.Subscript(value=ast.Name(id='args', ctx=ast.Load()),
                                                                   slice=ast.Constant(value=i), ctx=ast.Load())))
    # send the variables to the update function of each threads
    update_call = ast.Expr(value=new_function_call(single_var + ".copy_to"))
    update_call.value.args.append(ast.Name(id=update_id, ctx=ast.Load()))

    # register the update function of each threads
    set_call = ast.Expr(new_function_call(single_var + ".copy_from"))
    set_call.value.args = [ast.Name(id=ctx.with_node.local_vars[arg], ctx=ast.Load()) for arg in args]
    body.append(set_call)

    ast.copy_location(set_call, ctx.with_node)

    return args, [update_function, update_call]
