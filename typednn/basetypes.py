# TODO: add hash for TYPES if necessary
# we can also support infering the auxiliary data information from the input data information; for example, the shape and dtypes.
import typing

def iterable(x):
    return isinstance(x, typing.Iterable)


class InstantiationFailure(Exception):
    """Error in type inference"""

def match_list(A, B: typing.List["Type"]):
    if len(A) != len(B):
        raise InstantiationFailure("length mismatch")
    outs = []
    for a, b in zip(A, B):
        outs.append(b.instance(a))
        if outs[-1] is None:
            raise InstantiationFailure("type mismatch")
    return outs


class Type:
    def __init__(self, type_name) -> None:
        self._type_name = type_name
        assert self.__class__ == Type, "Type should not be instantiated directly."

    def reinit(self, *children):
        # create new types based on the provided chidren
        return self.__class__(*children, **self._get_extra_info())

    def _get_extra_info(self):
        return {}

    @property
    def is_type_variable(self):
        return hasattr(self, '_type_name')

    def __eq__(self, other):
        return self.__repr__() == other.__repr__()

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._type_name

    def __hash__(self) -> int:
        return hash(self.__repr__())

    def update_name(self, fn) -> "Type":
        if not self.polymorphism:
            return self
        args = []
        
        if self.is_type_variable:
            args.append(fn(self._type_name))
        return self.reinit(*args, *[i.update_name(fn) for i in self.children()])

    def children(self) -> typing.Tuple["Type"]:
        return ()
    
    def match_many(self):
        return False

    def sample(self):
        raise NotImplementedError("sample is not implemented for type %s" % self.__class__.__name__)

    @property
    def polymorphism(self):
        #TODO: accelerate this ..
        if hasattr(self, '_polymorphism'):
            return self._polymorphism
        self._polymorphism = True
        if self.is_type_variable:
            return True
        for i in self.children():
            if i.polymorphism:
                return True
        self._polymorphism = False
        return False


    def _test_unify(self, *args):
        from .unification import unify, TypeInferenceFailure
        if self.is_type_variable:
            args = [self._type_name] + list(args)

        new_type = self.reinit(*args)
        try:
            out = unify(new_type, self, None)
            out = out[0]
        except TypeInferenceFailure:
            raise InstantiationFailure
        return out

    def instantiate_children(self, x) -> "Type":
        return []

    def instance(self, x):
        try:
            children = self.instantiate_children(x)
        except InstantiationFailure:
            return None

        for i in children:
            if i is None:
                return None

        out = self._test_unify(*children)
        return out

    def check_compatibility(self, other):
        # by default we must match the type class and the 
        from .unification import TypeInferenceFailure
        A = self.children()
        if len(A) == 0:
            if str(self) != str(other):
                raise TypeInferenceFailure(f"type {str(self)} is not compatible with type {str(other)}: type name does not match.")
            return ([], [])

        B = other.children()
        if len(A) != len(B):
            raise TypeInferenceFailure(f"type {str(self)} is not compatible with type {str(other)}: number of children does not match.")

        if not issubclass(other.__class__, self.__class__):
            raise TypeInferenceFailure(f"type {str(self)} is not compatible with type {str(other)}: type class does not match.")
        return (A, B)


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

    def instantiate_children(self, inps) -> "Type":
        if not iterable(inps):
            raise InstantiationFailure("input is not iterable")
            
        inps = list(inps)
        if self.dot is None and len(inps) != len(self.elements):
            raise InstantiationFailure("input length does not match")
        if self.dot is not None and len(inps) < len(self.elements)-1:
            raise InstantiationFailure("input length does not match")

        if self.dot is None:
            return match_list(inps, self.elements)
        else:
            l = self.dot
            r = len(self.elements) - l - 1
            A, B, C = [], [], []
            if l > 0:
                A = match_list(inps[:l], self.elements[:l])
            if r > 0:
                C = match_list(inps[-r:], self.elements[-r:])
            B = self.elements[l].instance(inps[l:-r])
            return A + B + C
    
    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return TupleType(*self.elements[idx])
        return self.elements[idx]

    def check_compatibility(self, other):
        def contain_many(type_list):
            s = sum([i.match_many() for i in type_list])
            assert s <= 1, "can not have two variable arguments."
            return s > 0
        from .unification import TypeInferenceFailure
        a_children = self.children()
        b_children = other.children()
        # return super().check_compatibility(other)

        class_compability = issubclass(other.__class__, self.__class__)
        if not class_compability:
            #return False, (None, None)
            raise TypeInferenceFailure(f"type {self} is not compatible with type {other}.")

        if not contain_many(a_children) and not contain_many(b_children):
            # no ..., directly match two lists
            if len(a_children) != len(b_children):
                raise TypeInferenceFailure(f"type {self} has a different number of children with type {other}.")
            return (a_children, b_children)
        else:
            if not contain_many(a_children):
                C, D, dir = b_children, a_children, 1
                dir = 1
            else:
                C, D, dir = a_children, b_children, 0

            A = []
            B = []
            def resolve(x, y, dir):
                if dir == 1:
                    x, y = y, x
                A.append(x)
                B.append(y)

            def match_prefix(C, D):
                idx = 0
                while True:
                    if C[idx].match_many() or D[idx].match_many():
                        C = C[idx:]
                        D = D[idx:]
                        break
                    resolve(C[idx], D[idx], dir)
                    idx += 1
                return C, D

            # we now remove the common prefix or suffix until meet a ... or the end
            C, D = match_prefix(C, D)
            C, D = match_prefix(C[::-1], D[::-1]); C, D = C[::-1], D[::-1]

            # make C start with ...
            if len(D) > 0 and D[0].match_many():
                C, D = D, C
                dir = 1 - dir
            assert C[0].match_many()

            def resolve_many(arg_types, types, dir):
                if arg_types.base_type is not None:
                    for i in types:
                        if not i.match_many(): # for case 1
                            resolve(arg_types.base_type, i, dir)
                resolve(arg_types, TupleType(*types), dir)
            # print(C, D, self, other)

            if len(D) > 0:
                if C[0].match_many() and D[0].match_many():
                    # case1: the most difficult case
                    # ... blabla
                    if len(C) > 1:
                        if len(D) != 1:
                            raise TypeInferenceFailure
                        C, D = D, C
                        dir = 1 - dir
                    # C is ...; D is ... blabla
                    resolve_many(C[0], D, dir)
                else:
                    if len(C) != 1:
                        raise TypeInferenceFailure
                    # case2: associate C[0] to the remaining element of D
                    resolve_many(C[0], D, dir)

            return (A, B)


class ListType(Type): # sequence of data type, add T before the batch
    def __init__(self, base_type: Type) -> None:
        self.base_type = base_type

    def __str__(self):
        return f"List({self.base_type})"

    def children(self):
        return (self.base_type,)

    def instantiate_children(self, x):
        if not (isinstance(x, list) or isinstance(x, tuple)):
            return None
        return [self.base_type.instance(i) for i in x]


class VariableArgs(Type): # is in fact the ListType with unknown length
    # something that can match arbitrary number of types
    def __init__(self, type_name, based_type: typing.Optional["Type"]):
        self._type_name = type_name
        self.base_type = based_type
        assert self.base_type is not None 

    def match_many(self):
        return True

    def __str__(self):
        if self.base_type is None:
            return self._type_name + "*"
        return self._type_name + "(" + str(self.base_type) + ")"

    def instantiate_children(self, x):
        if not iterable(x):
            raise InstantiationFailure(f"can not instantiate {self} with {x}.")
        return [self.base_type.instance(i) for i in x]

    def instance(self, x):
        from .unification import unify, TypeInferenceFailure
        try:
            children = self.instantiate_children(x)
        except InstantiationFailure:
            return None
        if len(children) == 0:
            return []
        try:
            base_type = self.base_type
            outs = []
            for i in children:
                i, base_type, _ = unify(i, base_type, None)
                outs.append(i)
            return outs
        except TypeInferenceFailure as e:
            return None

    def children(self):
        if self.base_type is not None:
            return (self.base_type,)
        return ()
    

        
class UnionType(Type):
    def __init__(self, *types) -> None:
        raise NotImplementedError

        
class Arrow(Type):
    def __init__(self, *args) -> None:
        self.args = args[:-1]
        self.out = args[-1]

    def __str__(self):
        return '->'.join(str(e) for e in self.children())

    def children(self) -> typing.Tuple["Type"]:
        return list(self.args) + [self.out]

    def unify(self, *args):
        from .unification import unify
        return unify(TupleType(*args), TupleType(*self.args), self.out)

    def test_unify(self, gt, *args):
        from .unification import TypeInferenceFailure
        print("testing unify", self)
        print("INPUT:")
        for i in args:
            print(" " + str(i))
        print("Output:")
        if gt != 'error':
            output = str(self.unify(*args)[-1])
            assert output == gt, "unify failed: " + output + " != " + gt
            print("unify succeed! ", output + ' == ' + gt)
            print("\n\n")
        else:
            try:
                self.unify(*args)
                assert False, "unify should fail!"
            except TypeInferenceFailure:
                print("unify failed as expected.")
                print("\n\n")

    def instance(self, x):
        raise NotImplementedError("Arrow is not a simple type, it's a function type.")

        
        
class AttrType(Type):
    # type that supports attributes
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        #self.type_name = type_name or "AttrType"
        

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        raise AttributeError(name)

    def children(self) -> typing.Tuple["Type"]:
        return tuple(self.kwargs.values())

    def reinit(self, *children):
        return self.__class__(**dict(zip(self.kwargs.keys(), children)))

    # def instance(self, x):
    #     for k, v in self.kwargs.items():
    #         if not hasattr(x, k):
    #             return False
    #         if not v.instance(getattr(x, k)):
    #             return False
    #     return True

    def instantiate_children(self, x) -> "Type":
        outs = []
        for k, v in self.kwargs.items():
            if not hasattr(x, k):
                raise InstantiationFailure(f"can not instantiate {self} with {x}.")
            outs.append(v.instance(getattr(x, k)))
        return outs
    
    def __str__(self):
        return self.__class__.__name__ + "(" + ", ".join(f"{k}={v}" for k, v in self.kwargs.items()) + ")"

    def check_compatibility(self, other):
        # we just needs all subclasses to have the same compatible attributes
        A, B = [], []
        for k, v in self.kwargs.items():
            if not hasattr(other, k):
                return False, (None, None)
            A.append(v)
            B.append(getattr(other, k))
        return issubclass(other.__class__, self.__class__), (A, B)


class DataType(AttrType):
    # data_cls could be anything ..
    def __init__(self, data_cls, type_name=None):
        self.data_cls = data_cls
        self.type_name = type_name or self.data_cls.__name__

    def __str__(self):
        #return self.data_cls.__name__
        return self.type_name

    def instance(self, x):
        if isinstance(x, self.data_cls):
            return self
        return None

    def children(self):
        return ()


class PType(Type):
    # probablistic distribution of the base_type
    def __init__(self, base_type) -> None:
        raise NotImplementedError
