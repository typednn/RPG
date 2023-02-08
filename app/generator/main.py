from tools.nn_base import Network


class Generator(Network):
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def create_from_types(self, input_types, output_types):
        pass

    def inference(self, *inputs):
        raise NotImplementedError

    def training(self, *inputs):
        raise NotImplementedError