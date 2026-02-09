import ast
from typing import cast

from omp4py.core.parser.tree import Parallel
from omp4py.core.preprocessor.transformers.transformer import Context, construct

@construct.register
def _(ctr: Parallel, body: list[ast.stmt], ctx: Context) -> list[ast.stmt]:


    return body
