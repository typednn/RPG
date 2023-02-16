from typednn import Class
from typednn.types import TensorType
from .camera import Camera

class MultiViewImages(Class):
    image: TensorType('...', 3, 'H', 'W')
    #camera: Camera('...')
    mask: TensorType('...', 'H', 'W')
    aabb: TensorType('...', 2, 2)

    #@torchop
    def pixel2rgb(self, pixels: TensorType('...', 'B', 2)) -> TensorType('...', 'B', 3):
        pass