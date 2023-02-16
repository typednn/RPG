from typednn import Class
from typednn.types import AttrType, TensorType

class Field(Class):
    # the encoder can be either a pytorch operator or a neural network
    def encode(self, coords: TensorType('B', 'N', 3)) -> TensorType('B', 'N', 3):
        pass
    
# def build(data_type: [image, tensor], condition_type: [image, tensor], model_name=None, **kwargs):
#     if model_name is not None:
#         return xxx

#     vector & vector
#     if data_type == 'image' and condition_type == 'tensor':
#         return ImageTensorConditionUnet(**kwargs)
        

class TensorF(Field):
    # plane_coef = torch.nn.Parameter(
    #     0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, res), device=device))
    # line_coef = torch.nn.Parameter(
    #     0.1 * torch.randn((3, self.app_n_comp + self.density_n_comp, res, 1), device=device))
    # basis_mat = torch.nn.Linear(self.app_n_comp * 3, self.app_dim, bias=False, device=device)
    _config = {'app_n_comp': 3, 'density_n_comp': 3, 'res': 64, 'app_dim': 10}

    plane_coef: TensorType('...', 3, _config['app_n_comp'] + _config['density_n_comp'] , _config['res'], _config['res'])
    line_coef: TensorType('...', 3, _config['app_n_comp'] + _config['density_n_comp'], '1')
    basis_mat: TensorType('...', _config['app_n_comp'] * 3, _config['app_dim'])

    @asmethod
    def encode(self, coords: TensorType('...', 'N', 3)) -> TensorType('...', 'N', 3):
        return super().encode(coords)

        
    @pyfunc
    def additional_losses() -> TensorType('...'):
        pass