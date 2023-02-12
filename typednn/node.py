# computation node
# TODO: structured node (DictNode)
# TODO: ArrowNode -> generate (and init) operator if necessary

import abc
import typing
import copy
from .basetypes import Type, TupleType, AttrType, Arrow


def nodes_to_types(nodes):
    from .node import Node
    nodes: typing.List[Node]
    return [i.get_type() for i in nodes]


def nodes_to_metatype(nodes):
    from .node import Node
    nodes: typing.List[Node]
    return [i._meta_type for i in nodes]


# global NODEID
NODEID = 0


class NodeBase(abc.ABC):
    @classmethod
    def from_val(cls, val):
        if isinstance(val, NodeBase):
            return val

        from .operator import Operator
        if isinstance(val, Type):
            return InputNode(val)
        else:
            #return ValNode(val)
            raise NotImplementedError("not supported")

    @abc.abstractmethod
    def get_type(self):
        pass

    def __init__(self, name=None) -> None:
        super().__init__()
        self._name = name
        self._meta_type = None

        global NODEID
        self._id = NODEID
        NODEID += 1

    def get_parents(self):
        # find all nodes that are connected to this node
        return []

    def __str__(self) -> str:
        return f'{self.__class__.__name__}({self.get_type()})'

    def __hash__(self) -> int:
        return hash(f'THISISANODEWITHID:{self._id}')

    def __copy__(self):
        raise NotImplementedError("copy is not supported for Nodes")

    def __deepcopy__(self, memo):
        deepcopy_method = self.__deepcopy__
        self.__deepcopy__ = None
        #node = super().__deepcopy__(memo)
        node = copy.deepcopy(self)
        self.__deepcopy__ = deepcopy_method
        node.__deepcopy__ = deepcopy_method

        global NODEID
        node._id = NODEID
        NODEID += 1
        return node

    @abc.abstractmethod
    def evaluate(self, context):
        pass


class InputNode(NodeBase):
    # TODO: remove input node ...
    def __init__(self, type, **kwargs) -> None:
        super().__init__(**kwargs)
        self._meta_type = self._type = type

    def get_type(self):
        return self._type

    def evaluate(self, context):
        return context[self]

    def __str__(self) -> str:
        out = super().__str__()
        if self._name is not None:
            return f'{self._name}:{out}'
        return out


# class ValNode(NodeBase):
#     def __init__(self, val, **kwargs) -> None:
#         super().__init__(**kwargs)
#         self._meta_type = self.val = val

#     def get_type(self):
#         return self.val

#     def evaluate(self, context):
#         return self.val


class Node(NodeBase): # compile a static type tree based on meta type 
    def __init__(self, meta_type, **kwargs) -> None:
        super().__init__(**kwargs)
        self._meta_type = meta_type
        self._type = None

    @abc.abstractmethod
    def get_parents(self):
        pass

    def evaluate(self, context):
        if self not in context:
            context[self] = self._evaluate(context)
        return context[self]

    @abc.abstractmethod
    def _evaluate(self, context):
        pass

    @abc.abstractmethod
    def _get_type(self):
        pass

    @abc.abstractmethod
    def print_line(self):
        pass

    def get_type(self):
        if self._type is None:
            self._type = self._get_type()
        return self._type

    def __iter__(self):
        if not isinstance(self._meta_type, TupleType):
            raise RuntimeError(f"Only TupleType or its subclass can iterate, but get {self}.")

        for i in range(len(self._meta_type)):
            if ',' in self._name:
                name = str(self._name)[1:-1].split(',')[i]
            else:
                name = self._name + f'.{i}'
            yield IndexNode(self._meta_type[i], self, index=i, name=name)
            
    # def __getattr__(self, key) -> str:
    #     if not isinstance(self._meta_type, AttrType):
    #         raise RuntimeError(f"Only AttrType or its subclass can have attributes, but get {self._meta_type}.")
    #     if not hasattr(self._meta_type, key):
    #         raise RuntimeError(f"{self} does not have attribute {key}.")
    #     return AttrNode(getattr(self._meta_type, key), self, key=key, name=self._name + f".{key}", )

    def compile(self, *args, **kwargs):
        from .compiler import compile
        return compile(self, *args, **kwargs)

    def call_partial(self, *args, **kwargs):
        #raise NotIM
        raise NotImplementedError


class ShadowNode(Node): 
    # input node of the operator 
    # hack so that we can modify the default input node easily during reconfiguration
    def __init__(self, type, **kwargs) -> None:
        Node.__init__(self, type._meta_type, **kwargs)
        self._parent_type = type

    def get_type(self):
        return self._parent_type.get_type()
        
    def _evaluate(self, context):
        return self._parent_type.evaluate(context)

    def __str__(self) -> str:
        return 'INP:' + self._parent_type.__str__()

            
            
class IndexNode(Node):
    def __init__(self, meta_type, parent, index, **kwargs) -> None:
        super().__init__(meta_type, **kwargs)
        self.parent: Node = parent
        self.index = index
    
    def get_parents(self):
        return [self.parent]

    def _evaluate(self, context):
        return self.parent.evaluate(context)[self.index]

    def _get_type(self):
        return self.parent.get_type()[self.index]


    def print_line(self):
        return str(self.parent._name) + '[' + str(self.index) + ']'

            
class AttrNode(Node):
    def __init__(self, meta_type, parent, key, **kwargs) -> None:
        super().__init__(meta_type, **kwargs)
        self.parent: Node = parent
        self.key = key

    def get_parents(self):
        return [self.parent]
    
    def _evaluate(self, context):
        return getattr(self.parent.evaluate(context), self.key)

    def _get_type(self):
        return getattr(self.parent.get_type(), self.key)

    def print_line(self):
        return str(self.parent._name) + '.' + str(self.key)

    def __getattr__(self, key) -> str:
        if not isinstance(self._meta_type, AttrType):
            raise RuntimeError(f"Only AttrType or its subclass can have attributes, but get {self._meta_type}.")
        if not hasattr(self._meta_type, key):
            raise RuntimeError(f"{self} does not have attribute {key}.")
        return AttrNode(getattr(self._meta_type, key), self, key=key, name=self._name + f".{key}", )
        

from .application import CallNode



if __name__ == '__main__':
    node = InputNode(1)
    import copy
    node2 = copy.deepcopy(node)
    node2 = copy.deepcopy(node)
    assert hash(node2) != hash(node)
    print(node._id)
    print(node2._id)
    print(NODEID)