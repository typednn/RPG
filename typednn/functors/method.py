"""
When we define the graph, taking a method attribute of a node will return the corresponding method wrapper.


Note that we require the type is annotated ..
"""
#from .funcdef import moduledef
from ..basetypes import AttrType
from ..functor import Code


REGISTERED_OPERATORS = {}


def asop(inherit=False):
    #TODO: determine the type to use
    assert not inherit, "if inherit is True, which means methods of the subclass will reuse the same operator, which is not supported yet."

    def decorator(method):
        def wrapper(*args, **kwargs):
            self = args[0]

            name = str(self)
            if name in REGISTERED_OPERATORS:
                op = REGISTERED_OPERATORS[name]['op']
                return op.shadow(*args, **kwargs)
            else:
                op = method(*args, **kwargs)
                # create input nodes ..
                Args = [DetachNode(i) for i in args]
                REGISTERED_OPERATORS[name] = {
                    'op': op
                }
                return op.get_output()
        return wrapper
    return decorator


def test():
    from ..types import TensorType, MLP
    class NewType(AttrType):
        a: TensorType('B', 'N')

        @asop(inherit=False)
        def encode(self):
            return MLP(self.a)
    
    new_type = NewType(a=TensorType('B', 100))
    print(new_type.encode())

    #encode = new_type.encode()
    #print(encode)

    #inp = new_type.sample()
    #inp.encode()
    new_type2 = NewType(TensorType(512, 100))

    inp = new_type2.sample()
    #print(inp['encode'])
    output_node = inp.encode()
    print(output_node.get_type())
    exit(0)
    out = inp.encode()
    print(type(out))
    exit(0)
    #print(out.shape)



if __name__ == '__main__':
    test()