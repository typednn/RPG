class AttrDict(dict):
    def __init__(self, *args, _base_type=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._base_type = _base_type
    def __getattr__(self, key):
        #if self._base_type is not None and hasattr(self._base_type, key):
        #    return getattr(self._base_type, key)
        return self[key]
    def __setattr__(self, key, val):
        self[key] = val
    def __delattr__(self, key):
        del self[key]

    def __str__(self) -> str:
        out = super().__str__()
        return out

    def visit(self, func):
        out = {}
        for key, val in self.items():
            if isinstance(val, AttrDict):
                v = val.visit(func)
            else:
                v = func(val)
            out[key] = v
        return out

    @property
    def shape(self):
        import torch
        def get_shape(x):
            if isinstance(x, torch.Tensor):
                return x.shape
            return None
        return self.visit(get_shape)