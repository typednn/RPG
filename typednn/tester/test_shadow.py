import torch
from ..types import TensorType, MLP
from ..functors import Tuple


def test_shadow():
    inp = TensorType('...', 100)

    mlp = MLP(inp, layer=2, hidden=512, out_dim=100)
    out = mlp.reuse(mlp)

    repeat = Tuple(mlp, out).compile()

    mlp2 = MLP(inp, layer=2, hidden=512, out_dim=100)

    mlp.init()
    mlp2.init()

    mlp2.code.load_state_dict(mlp.code.state_dict())

    
    i = inp.sample()

    assert torch.allclose(repeat.forward(i)[0], mlp2(i))
    assert torch.allclose(repeat.forward(i)[1], mlp2(mlp2(i)))

    
    
if __name__ == '__main__':
    test_shadow()