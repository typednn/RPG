# https://github.com/hzaskywalker/TaskAnnotator/blob/main/llm/pl/unification.py
#from .types import TypeInferenceFailure, Arrow, List, Type, Tuple
"""
TODO: support UNION type
"""
import typing 
from .basetypes import TupleType, Type, VariableArgs


class TypeInferenceFailure(Exception):
    """Error in type inference"""


def map_type(tp: Type, f):
    if len(tp.children()) > 0:
        return tp.reinit([map_type(i, f) for i in tp.children()])
    else:
        return f(tp)

def check_occurs(a, b):
    if str(a) == str(b):
        return True
    for i in b.children():
        if check_occurs(a, i):
            return True
    return False


def check_compatibility(a: Type, b: Type, dir):
    if dir:
        a, b = b, a
    B, A = b.check_compatibility(a)
    if dir:
        A, B = B, A
    return A, B
    

def unify(
    tpA: Type,
    tpB: Type,
    query: Type,
    update_name=True,
    queryA=False,
):
    # https://en.wikipedia.org/wiki/Hindley%E2%80%93Milner_type_system

    if isinstance(tpA, list):
        tpA = TupleType(*tpA)
    if isinstance(tpB, list):
        tpB = TupleType(*tpB)
    if query is not None and isinstance(query, list):
        query = TupleType(*query)

    if update_name:
        tpB = tpB.update_name(lambda x: x + '\'')
        if query is not None and not queryA:
            query = query.update_name(lambda x: x + '\'')

    pa = {} # map str to type in the end ..
    def findp(a: Type):
        if a.is_type_variable:
            s = str(a)
            if s not in pa:
                pa[s] = a
            if pa[s] != a:
                pa[s] = findp(pa[s])
            return pa[s]
        else:
            return a

    def resolve(a, b, dir):
        # if dir (direction) is 1, we want to unify b into a; so we call b to check compatibility with a
        error = None
        try: 
            a = findp(a)
            b = findp(b)
            if a.is_type_variable:
                if str(a) != str(b):
                    if check_occurs(a, b):
                        raise TypeInferenceFailure("Recursive type ..")
                    pa[str(a)] = b
            elif b.is_type_variable:
                resolve(b, a, dir=1-dir)
            else:
                (a_children, b_children) = check_compatibility(a, b, dir)
                for x, y in zip(a_children, b_children):
                    # print('children', 'x', x, 'y', y, 'a', a,'b',  b)
                    resolve(x, y, dir) 
        except TypeInferenceFailure as e:
            error = TypeInferenceFailure(str(e) + f" when unifying {a} and {b}\n")

        if error is not None:
            raise error


    resolve(tpA, tpB, 0)

    allocator = {}
    # we need to allocate them with names in A
    for i in pa:
        p = findp(pa[i])
        if not i.endswith('\''):
            if p.is_type_variable and p._type_name not in allocator and p._type_name.endswith('\''):
                allocator[p._type_name] = i

    def substitute(x: Type):
        if not x.polymorphism: # no type variable, directly return
            return x

        if not x.is_type_variable and len(x.children()) > 0:
            # can't be type variable
            out = []
            for i in x.children():
                if isinstance(i, VariableArgs):
                    y = substitute(i)
                    if isinstance(y, TupleType):
                        out += y.children()
                    else:
                        out.append(y)
                else:
                    out.append(substitute(i))
            return x.reinit(*out)

        assert x.is_type_variable
        p = findp(x)
        if p.is_type_variable:
            return allocator[p._type_name] # Currently we don't allow any name in B not in A .. this should be removed later ..
        else:
            return p

    tpA = substitute(tpA)
    tpB = substitute(tpB)
    if query is not None:
        query = substitute(query)
    return tpA, tpB, query


def test_inheritance():
    raise NotImplementedError
    

if __name__ == '__main__':
    test_inheritance()