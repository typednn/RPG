import torch
from ...dataset import Element, int_type, dataset_type
from torchvision import datasets, transforms
from ... import DATA_PATH
from typednn.types import AttrType, TensorType
from typednn import types, AttrDict


class MNISTDataset(Element, torch.utils.data.Dataset):
    arrow = dataset_type(TensorType(1, 28, 28, data_dims=3), TensorType(1, dtype=torch.long), int_type)

    def _build_dataset(self):
        self.transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return datasets.MNIST(DATA_PATH, train=self.config.train, download=True, transform=transforms.ToTensor())

    def __getitem__(self, index):
        self.init()
        img, out = self.main[index]
        return AttrDict(input=img, output=torch.tensor([out], dtype=torch.long))

        
if __name__ == '__main__':
    mnist = MNISTDataset(train=True)
    print(mnist.arrow)
    mnist.reconfig(train=False)
    print(len(mnist))
    print(mnist(10).input.shape)
    print(mnist(10).output.shape)