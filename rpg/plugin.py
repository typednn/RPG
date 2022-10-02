from tools.config import Configurable

class Plugin(Configurable):
    def __init__(self, cfg=None):
        pass

    def return_default(self, x, buffer):
        if x is None:
            return x
        return buffer

    def update_data(self, data, locals_):
        pass

    def _update_data(self, data, locals_):
        return self.return_default(
            self._update_data(data, locals_), data)


    def on_transition(self, buffer, **local):
        pass

    # pass

    def on_batch_learning(self):
        pass

    def on_training(self):
        pass
