import torch
from ..types import TensorType, MLP
from ..functors import Tuple


def test_shadow():
    inp = TensorType('...', 100)

    mlp = MLP(inp, layer=2, hidden=512, out_dim=100)
    out = mlp.shadow(mlp)

    repeat = Tuple(mlp, out).compile()

    mlp2 = MLP(inp, layer=2, hidden=512, out_dim=100)
    mlp2.load_state_dict(mlp.state_dict())

    
    i = inp.sample()

    assert torch.allclose(repeat(i)[0], mlp2(i))
    assert torch.allclose(repeat(i)[1], mlp2(mlp2(i)))

    
    
if __name__ == '__main__':
    test_shadow()