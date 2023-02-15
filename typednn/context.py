from .node import Node

class Visitor(dict):
    def __init__(self, key, context):
        self.dict = {}
        self.key = key
        self.context = context

    def visit(self, node: Node):
        if node in self.dict:
            return self.dict[node]
        outs = []
        for p in node.get_parents():
            outs.append(self.visit(p))

        out = getattr(node, f'_get_{self.key}')(
            *outs, context=self.context)
        self.dict[node] = out
        return out

    def __getitem__(self, node: Node):
        return self.dict[node]

class ContextManager:
    def __init__(self) -> None:
        self.config = Visitor('config') # configuration of the node
        self.module = Visitor('module') # callable pytorch modules
        self.type = Visitor('type') # type of the node
        self.evaluate = Visitor('evaluate') # evaluated value of the node