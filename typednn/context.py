"""
We use context and visitor to store the execution results of the nodes. 
It includes:
    - config
    - module 
    are the attributes of the node.

    - type
    - evaluation results 
    are the results of the node.

Context unifies the code for all attributes.

Each node has only one context.
However, each context can have multiple subcontexts for functions reused at different places.
"""
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
        self.dict[node] = out
        return out

    def __getitem__(self, node: Node):
        return self.visit(node)

ContextID = 0

class Context:
    def __init__(self, name=None) -> None:
        #self.config = Visitor('config', self) # configuration of the node
        #self.module = Visitor('module', self) # callable pytorch modules
        self.type = Visitor('type', self) # type of the node
        self.evaluate = Visitor('evaluate', self) # evaluated value of the node

        self.name = name
        self.applications = {}
        self.children = []

        global ContextID
        self.ID = ContextID
        ContextID += 1

    def store_application(self, caller):
        out = self.applications.get(caller.op, [])
        out.append(caller)
        self.applications[caller.op] = out

    def add_subcontext(self, context):
        self.children.append(context)

    def initialized(self, node):
        return node in self.module.dict

    def __hash__(self) -> int:
        return hash(f'THISISACONTEXTWITHID:{self.ID}')


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