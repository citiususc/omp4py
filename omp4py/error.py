import ast
from typing import Union


class OmpError(Exception):
    pass


class OmpSyntaxError(SyntaxError):

    def __init__(self, msg: Union[str, 'OmpSyntaxError'] = None, filename: str = None, node: ast.AST = None):
        if isinstance(msg, OmpSyntaxError):
            SyntaxError.__init__(self, msg.msg, (msg.filename, msg.lineno, 0, msg.text))
        else:
            sample = ast.unparse(node).split("\n")[0]
            SyntaxError.__init__(self, msg, (filename, node.lineno, 0, sample))
