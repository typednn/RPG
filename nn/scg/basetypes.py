"""
type system of neural networks
 a type is actually similar to the gym Box
 typed_builder is a builder that is able to select the default config based on the type of the input and output 

 the major use case is to specify the type of the input and output of the neural network 
 ideally, when we specify the type of the data, we can automatically generate the neural network based on some high-level specification
   instead of choosing the networks by ourselves.
   for example: we can define a seq(a) -> seq(a) network, and the type of the input and output is seq, then we can automatically generate the network based on the element of the seq(a)

 another things to choose is that if we need to use the dynamic types? Infer the input and select the result? 


 abstract operators for datas:
   - stack (generate sequence)
   - UNet (transformation)
   - batchify a sequence (sparse, or dense mode)
   - reshape a batch of data back in to packed sequences
   - Encoding, Decoding
   - reshape (only reshape the batch dimension)
  Also, we can support different ways of fusion different things. 


 for probabilistic distributions, we have the operators
   - sample (size, ...) -> data, logp, entropy
   - rsample if possible (size, ...)
   - log_prob (evaluate the log prob of the data, ...) (if supported)

 we build the stochastic computation graph for all elements
  - we can sample a element by sampling its all ancestors
  - we can condition a element by conditioning its all ancestors
  - one can evaluate the log prob of a element (together with conditions and sampled results) by evaluating the log prob of its all sampled ancestors
"""

# we can also support infering the auxiliary data information from the input data information; for example, the shape and dtypes.

class Type:
    def __init__(self, type_name) -> None:
        self._type_name = type_name

    @property
    def is_type_variable(self):
        return hasattr(self, '_type_variable') and self._type_name.startswith('\'')

    def __eq__(self, other):
        return str(self) == str(other)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self._type_name

    @property
    def children(self):
        return ()
    
    def match_many(self):
        return False

    @property
    def polymorphism(self):
        #TODO: accelerate this ..
        if len(self.children) > 0:
            for i in self.children:
                if i.polymorphism:
                    return True
            return False
        else:
            return hasattr(self, '_type_name')


class DataType(Type):
    # data_cls could be anything ..
    def __init__(self, data_cls, type_name=None):
        self.data_cls = data_cls
        self.type_name = type_name or self.data_cls.__name__

    def __str__(self):
        #return self.data_cls.__name__
        return self.type_name

    def instance(self, x):
        return isinstance(x, self.data_cls)

    @property
    def children(self):
        return ()
        

class TupleType(Type):
    # can be named later
    def __init__(self, *args, **kwargs) -> None:
        assert len(args) is None or kwargs is None

        
class TensorType(Type):
    # multi-dimenional array
    def __init__(self, type_name) -> None:
        super().__init__(type_name)
    

class PType(Type):
    # probablistic distribution of the base_type
    def __init__(self, base_type) -> None:
        super().__init__()


class ListType(Type): # sequence of data type, add T before the batch
    def __init__(self, type_name) -> None:
        super().__init__(type_name)