import abc
import inspect
import typing
import copy
from .utils import frame_assert, exception_with_traceback
from .basetypes import Type, TupleType, AttrType, Arrow


def nodes_to_metatype(nodes):
    from .node import Node
    nodes: typing.List[Node]
    return [i._meta_type for i in nodes]

NODEID = 0

class Node(abc.ABC): # compile a static type tree based on meta type 
    trace_key = None

    def __init__(self, meta_type, name=None, trace=None) -> None:
        from .context import get_context
        self.context = get_context() # store where the node is created

        if self.trace_key is not None:
            self._name, self.trace = self.find_caller(self.trace_key, trace)
        else:
            self._name, self.trace = name, trace
        self._meta_type = meta_type
        global NODEID
        self._id = NODEID
        NODEID += 1

    @classmethod
    def from_val(cls, val):
        if isinstance(val, Node):
            return val

        from .operator import Code
        if isinstance(val, Type):
            return InputNode(val)
        else:
            #return ValNode(val)
            raise NotImplementedError(f"not supported input type for Node: {type(val)}")

    def get_type(self):
        return self.context.type[self]

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


    def myassert(self, cond, msg='', errorType=ValueError):
        #frame_assert(cond, msg, self.get_trace, errorType)
        frame_assert(cond, msg, lambda: self.trace or '', errorType)

    def find_caller(self, key, trace):
        key = key or 'OPERATORS'
        for frame in inspect.stack():
            if (key in frame[4][0]):
                frame = frame[0]
                code = inspect.getframeinfo(frame).code_context[0].strip()
                if code.startswith('return '):
                    name = 'return'
                else:
                    name = code.split("=")[0].strip()
                trace = ('\n' + trace if trace is not None else '') + '\n\ninit at\n' + exception_with_traceback(frame) 
                return name, trace
        raise ValueError("cannot find the caller of this function")

    @abc.abstractmethod
    def get_parents(self):
        pass

    @abc.abstractmethod
    def print_line(self):
        pass

    def _get_config(self, *args, context=None): # by default we don't config this ..
        return None

    def _get_module(self, *args, context=None):
        return None

    @abc.abstractmethod
    def _get_type(self, *args, context=None):
        pass

    @abc.abstractmethod
    def _get_evaluate(self, *args, context=None):
        pass


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
    #         raise RuntimeError(f"Missing key {key}: notice that only AttrType or its subclass can have attributes, but get {self._meta_type}.")
    #     if not hasattr(self._meta_type, key):
    #         raise RuntimeError(f"{self} does not have attribute {key}.")
    #     return AttrNode(getattr(self._meta_type, key), self, key=key, name=self._name + f".{key}", )

    def compile(self, *args, **kwargs):
        from .abstraction import abstract
        return abstract(self, *args, **kwargs)


class InputNode(Node):
    # TODO: remove input node ...
    def __init__(self, type, **kwargs) -> None:
        super().__init__(type, **kwargs)
        self._meta_type = self._type = type

    def _get_type(self, context=None):
        return self._type

    def _get_evaluate(self, context):
        return context[self]

    def get_parents(self):
        return []

    def print_line(self):
        return f'INP:{self._name}'

    def __str__(self) -> str:
        out = super().__str__()
        if self._name is not None:
            return f'{self._name}:{out}'
        return out

            
class IndexNode(Node):
    def __init__(self, meta_type, parent, index, **kwargs) -> None:
        super().__init__(meta_type, **kwargs)
        self.parent: Node = parent
        self.index = index
    
    def get_parents(self):
        return [self.parent]

    def _get_evaluate(self, parent, context):
        return parent[self.index]

    def _get_type(self, parent, context):
        return parent[self.index]

    def print_line(self):
        return str(self.parent._name) + '[' + str(self.index) + ']'

            
class AttrNode(Node):
    def __init__(self, meta_type, parent, key, **kwargs) -> None:
        super().__init__(meta_type, **kwargs)
        self.parent: Node = parent
        self.key = key

    def get_parents(self):
        return [self.parent]
    
    def _get_evaluate(self, parent, context):
        return parent[self.key]

    def _get_type(self, parent, context):
        return getattr(parent, self.key)

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
