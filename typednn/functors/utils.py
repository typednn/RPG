# utility function like concat, stack and so on ..
from ..operator import Operator

#class Concat(Operator):
#    def type_inference(self, *args):
#        self._oup_type = args[-1].out

class Seq:
    def __init__(self, *args) -> None:
        pass

    def type_inference(self, *args, **kwargs):
        pass