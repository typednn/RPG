from torch.nn import Module
# pytorch module, but typed
# shapes that we support:
#   - (num1, num2, num3)
#   - one ellipse (num1, ..., num2, num3)
#   - (num1, num2, IMG_TYPE)
#   - each num can be '?' that matches any other shapes or a type symbol



class Operator(Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @property
    def out(self): # out type when input are feed ..
        pass

    def _get_type(self):
        if not hasattr(self, '_type'):
            self._type = self.get_type()
        return self._type

    def get_type(self):
        raise NotImplementedError
