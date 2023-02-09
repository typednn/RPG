# Functor takes in modules as input and wrap them into a new module.
# This does not mean the module will be of 
from .operator import Operator


class Functor(Operator):
    #raise NotImplementedError
    #pass
    def __init__(self, *args, name=None, _trace_history=None, **kwargs) -> None:
        super().__init__(*args, name=name, _trace_history=_trace_history, **kwargs)


if __name__ == '__main__':
    pass