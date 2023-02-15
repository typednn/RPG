from .node import Node


class Visitor:
    def __init__(self, key, context):
        self.dict = {}
        self.key = key
        self.context = context

    def visit(self, node: Node):
        if node in self.dict:
            return self.dict[node]
        self.dict[node] = None # prevent infinite recursion
        outs = []
        for p in node.get_parents():
            outs.append(self[p])
        out = getattr(node, f'_get_{self.key}')(
            *outs, context=self.context)
        if self.key == 'config':
            print(node._name, out)
        self.dict[node] = out
        return out

    def __getitem__(self, node: Node):
        return self.visit(node)


class Context:
    def __init__(self, name=None) -> None:
        self.config = Visitor('config', self) # configuration of the node
        self.module = Visitor('module', self) # callable pytorch modules
        self.type = Visitor('type', self) # type of the node
        self.evaluate = Visitor('evaluate', self) # evaluated value of the node
        self.children = []

    def add_subcontext(self, context):
        self.children.append(context)

    def initialized(self, node):
        return node in self.module.dict


#TODO: add context manager/scope
#DEFAULT_CONTEXT = Context()
context_stack = [Context()]

def get_context() -> Context:
    return context_stack[-1]

class Scope:
    def __enter__(self, name=None, *args, **kwargs) -> Context:
        # args, kwargs are input nodes of the scope
        self.name = name
        self.layer_count = len(context_stack)
        context_stack.append(Context(name))
        self.context = context_stack[-1]
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        context_stack.pop()