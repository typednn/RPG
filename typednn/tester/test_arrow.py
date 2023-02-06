# named arrow
from ..unification import unify
from ..basetypes import *

def test():
    arrow = Arrow(A=Type("a"), B=Type("b"), out=Type('b'))
    b = Arrow(Arrow(Type("c"), Type("b")), Type('c'), NameType('c'))
    print(unify(b, arrow, None)[0])

    arrow2 = Arrow(VariableArgs('...', None), VariableArgs('...', None))
    b = Arrow(VariableArgs('...', None), NameType('C'))

    print(unify(arrow2, b, None))

if __name__ == '__main__':
    test()