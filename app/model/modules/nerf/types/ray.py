import torch
from .camera import TensorType, Camera, Class, torchop


class Ray(Class):
    origin: TensorType('...',  3)
    dir: TensorType('...',  3)
    index: TensorType('...',  1, dtype=torch.long)
    camera: Camera('...cam')

    @torchop
    def sample():
        pass

    @torchop
    def integration(self, ) -> TensorType('...', 3):
        pass