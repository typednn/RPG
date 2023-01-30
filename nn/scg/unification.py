# https://github.com/hzaskywalker/TaskAnnotator/blob/main/llm/pl/unification.py
#from .types import TypeInferenceFailure, Arrow, List, Type, Tuple
from .basetypes import TupleType


class TypeInferenceFailure(Exception):
    """Error in type inference"""


def map_type(tp, f):
    if len(tp.children) > 0:
        return tp.__class__(*[map_type(i, f) for i in tp.children])
    else:
        return f(tp)

def check_occurs(a, b):
    if str(a) == str(b):
        return True
    for i in b.children:
        if check_occurs(a, i):
            return True
    return False

    
TID = 0


def unify(tpA, tpB, queries):
    def update_name(tp):
        if not hasattr(tp, '_type_name'):
            return tp
        return tp.__class__(tp._type_name + '__right__')

    tpB = map_type(tpB, update_name)

    pa = {} # map str to type in the end ..

    def findp(a):
        #if len(a.children) > 0:
        #    return a.__class__(*[findp(i) for i in a.children])
        if a.is_type_variable:
            s = str(a)
            if s not in pa:
                pa[s] = a
            if pa[s] != a:
                pa[s] = findp(pa[s])
            return pa[s]
        else:
            return a

    def resolve(a, b):
        a = findp(a)
        b = findp(b)
        if a.is_type_variable:
            if str(a) != str(b):
                if check_occurs(a, b):
                    raise TypeInferenceFailure("Recursive type ..")
                pa[str(a)] = b
        elif b.is_type_variable:
            resolve(b, a)
        else:
            if len(a.children) == 0:
                if str(a) != str(b):
                    raise TypeInferenceFailure(f"type {a} != type {b}.")
                return
            def match(X, Y):
                if len(X) != len(Y):
                    raise TypeInferenceFailure(f"type {a} has a different number of children with type {b}.")
                for x, y in zip(X, Y):
                    resolve(x, y)
            def contain_many(X):
                s = sum([i.match_many() for i in X.children])
                assert s <= 1, "can not have two variable arguments."
                return s > 0

            if contain_many(b):
                a, b = b, a
            if not contain_many(a):
                match(a.children, b.children)
            else:
                C = a.children
                D = b.children
                def match_prefix(C, D):
                    idx = 0
                    while True:
                        if C[idx].match_many() or D[idx].match_many():
                            C = C[idx:]
                            D = D[idx:]
                            break
                        resolve(C[idx], D[idx])
                        idx += 1
                    return C, D

                C, D = match_prefix(C, D)
                C, D = match_prefix(C[::-1], D[::-1]); C, D = C[::-1], D[::-1]

                if len(D) > 0 and D[0].match_many():
                    C, D = D, C
                assert C[0].match_many()
                # assert len(C) == 1 or len(D) == 1, "can not match two variable arguments."

                def resolve2(D, C):
                    if len(C) > 0:
                        assert not C[0].match_many(), "can not match two variable arguments now."
                    if not D[0].is_type_variable:
                        for i in C:
                            resolve(D[0].base_type, i)
                    else:
                        resolve(D[0], TupleType(*C))

                if len(C) > 1:
                    assert len(D) == 1 and D[0].match_many()
                    resolve2(D, C)
                else:
                    resolve2(C, D)

    resolve(tpA, tpB)

    allocator = {}
    def substitute(x):
        if len(x.children) > 0:
            out = []
            for i in x.children:
                if i.match_many() and i.is_type_variable:
                    y = substitute(i).elements
                    out += y # requires this is a tuple
                else:
                    out.append(substitute(i))
            return x.__class__(*out)

        if not x.polymorphism:
            return x

        p = findp(x)
        #if is_type_variable(p):
        if p.is_type_variable:
            if p._type_name not in allocator:
                global TID
                allocator[p._type_name] = p.__class__(f'\'T{TID}')
                TID += 1

            out = allocator[p._type_name]
        else:
            out = substitute(p)
        allocator[x._type_name] = out
        return out

    substitute(tpA); substitute(tpB)


    queries = [update_name(i) for i in queries]
    return [substitute(i) for i in queries]




if __name__ == '__main__':
    pass