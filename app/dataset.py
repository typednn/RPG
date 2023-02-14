import torch
from typednn import Code, types
from typednn.application import DataNode
import typing
from abc import abstractmethod, ABC

int_type = types.UIntType("?")

def dataset_type(a, b, *args):
    return types.Arrow(*args, types.AttrType(input=a, output=b))
# class DatasetType(types.Type):
#     def __init__(self, input, output) -> None:
#         self.input = input
#         self.output = output

#     def __str__(self) -> str:
#         return f"DatasetType({self.input}, {self.output})"

#     def children(self) -> typing.Tuple["Type"]:
#         return [self.input, self.output]

class Dataset(Code):
    NODE_MAP = DataNode # add __len__ and __getitem__ to this class


class Element(Dataset): # Single data ..

    arrow = dataset_type(types.Type("A"), types.Type("B"), int_type)
    def _build_dataset(self):
        raise NotImplementedError("Must be implemented by subclass")

    def build_modules(self, input_idx):
        self.main = self._build_dataset()

    @classmethod
    def _new_config(cls):
        return dict(
            train=True
        )


    def __len__(self):
        return len(self.main)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def forward(self, idx):
        return self[idx]

    def __getitems__(self, idx):
        return [self[i] for i in idx]

        
class Batch(Code):
    def __init__(self, element, name=None, _trace_history=None, **kwargs) -> None:
        super().__init__(name=name, _trace_history=_trace_history, **kwargs)
        self.element = element

    @classmethod
    def _new_config(cls):
        return dict(
            batch_size=256,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )

    def build_modules(self):
        self.main = torch.utils.data.DataLoader(
            self.element, **self.config
        )
        self._iter = None

    def _type_inference(self):
        #return super()._get_type_from_output(output, *args)
        # add batch dimension ..
        def add_batch_dim(type):
            if isinstance(type, types.TensorType):
                return type.new(self.config.batch_size, *type.size)
            return type
        return self.element.arrow.out.map_types(add_batch_dim)

    def forward(self):
        if self._iter is None:
            self._iter = iter(self.main)
        try:
            out = next(self._iter)
        except StopIteration:
            self._iter = iter(self.main)
            out = next(self._iter)
        return out

        
if __name__ == '__main__':
    from app.model.datasets.mnist import MNISTDataset

    mnist = MNISTDataset(train=True)
    batched_minst = Batch(mnist)

    #print(batched_minst.arrow)
    out = batched_minst()
    print(out.input.shape)
    print(out.output.shape)