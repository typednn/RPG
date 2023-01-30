# we can also support infering the auxiliary data information from the input data information; for example, the shape and dtypes.
import typing

def match_list(A, B: typing.List["Type"]):
    if len(A) != len(B):
        return False
    for a, b in zip(A, B):
        if not b.instance(a):
            return False
    return True

def iterable(x):
    return isinstance(x, typing.Iterable)


class Type:
    def __init__(self, type_name) -> None:
        self._type_name = type_name

    @property
    def is_type_variable(self):
        return hasattr(self, '_type_variable') and self._type_name.startswith('\'')

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._type_name

    def update_name(self, fn) -> "Type":
        args = []
        if self.is_type_variable:
            args.append(fn(self._type_name))
        return self.__class__(*args, *map(fn, self.children))

    @property
    def children(self) -> typing.Tuple["Type"]:
        return ()
    
    def match_many(self):
        return False

    @property
    def polymorphism(self):
        #TODO: accelerate this ..
        if self.is_type_variable:
            return True
        for i in self.children:
            if i.polymorphism:
                return True
        return False

    def instance(self, x):
        return True


class TupleType(Type):
    # let's not consider named tuple for now ..
    def __init__(self, *args: typing.List[Type]) -> None:
        self.elements = []
        self.dot = None
        for idx, i in enumerate(args):
            if i.match_many():
                assert self.dot is None, NotImplementedError("only one ellipsis is allowed.")
                self.dot = idx
            self.elements.append(i)

    def __str__(self):
        return f"({', '.join(str(e) for e in self.children())})"

    def children(self):
        return tuple(self.elements) # + tuple(self.elements_kwargs.values())

    def instance(self, inps):
        #assert isinstance(inps, tuple) or isinstance(inps, list)
        if not iterable(inps):
            return False

        if self.dot is None and len(inps) != len(self.elements):
            return False
        if self.dot is not None and len(inps) < len(self.elements):
            return False

        if self.dot is None:
            return match_list(inps, self.elements)
        else:
            l = self.dot
            r = len(self.elements) - l
            if l > 0 and not match_list(inps[:l], self.elements[:l]):
                return False
            if r > 0 and not match_list(inps[-r:], self.elements[-r:]):
                return False
            return match_list(inps[l:-r], self.elements[l])



class ListType(Type): # sequence of data type, add T before the batch
    def __init__(self, base_type: Type) -> None:
        self.base_type = base_type

    def __str__(self):
        return f"List({self.base_type})"

    def children(self):
        return (self.base_type,)

    def instance(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            return False
        for i in x:
            if not self.base_type.instance(i):
                return False
        return True


class VariableArgs(Type):
    # something that can match arbitrary number of types
    def __init__(self, type_name, based_type: typing.Optional["Type"]=None):
        self._type_name = type_name
        self.base_type = based_type

    def match_many(self):
        return True

    def __str__(self):
        if self.base_type is None:
            return self._type_name + "(...)"
        return self._type_name + "(" + str(self.base_type) + ", ...)"

    def instance(self, x):
        if not iterable(x):
            return False
        if self.base_type is None:
            return True
        for i in x:
            if not self.base_type.instance(i):
                return False
        return True

    @property
    def children(self):
        if self.base_type is None:
            return (self.base_type,)
        return ()
    


class PType(Type):
    # probablistic distribution of the base_type
    def __init__(self, base_type) -> None:
        raise NotImplementedError


class DataType(Type):
    # data_cls could be anything ..
    def __init__(self, data_cls, type_name=None):
        self.data_cls = data_cls
        self.type_name = type_name or self.data_cls.__name__

    def __str__(self):
        #return self.data_cls.__name__
        return self.type_name

    def instance(self, x):
        return isinstance(x, self.data_cls)

    @property
    def children(self):
        return ()