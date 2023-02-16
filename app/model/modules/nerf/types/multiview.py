from typednn import Class

class MultiViewImages(Class):
    image: TensorType('...', 3, 'H', 'W')
    camera: Camera('...')
    mask: TensorType('...', 'H', 'W')
    aabb: TensorType('...', 4)

    @torchop
    def pixel2rgb(self, pixels: TensorType('...', 'B', 2)) -> TensorType('...', 'B', 3):
        pass