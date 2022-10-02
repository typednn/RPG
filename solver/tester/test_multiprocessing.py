def main():
    import torch
    from tools.optim import OptimModule

    class SGD(OptimModule):
        def __init__(
            # maxlen is the length of the buffer
            self, network, cfg=None,
            lr=3e-4, max_grad_norm=None, eps=1e-8, loss_weight=None, verbose=True,
            training_iter=1, batch_size=128, maxlen=3000,
            accumulate_grad=0,
        ):
            super().__init__(network, cfg)
            self.optimizer = torch.optim.SGD(self.params, lr=cfg.lr)


    from tools.dist_utils import rank_print, get_rank
    value = torch.nn.Parameter(
        torch.tensor(get_rank() + 1, dtype=torch.float32)
    )
    lossfn = lambda x: (x ** 2)
    opt = SGD(value, lr=0.3, accumulate_grad=2)

    for i in range(100):
        rank_print(f"iter:{i}")
        loss = lossfn(value)
        rank_print("before optimize")
        rank_print("\tgrad:", value.grad.item() if value.grad is not None else None)
        rank_print("\tvalue:", value.item())
        opt.optimize(loss)
        rank_print("after optimize")
        rank_print("\tgrad:", value.grad.item())
        rank_print("\tvalue:", value.item())

        
if __name__ == '__main__':
    from tools.dist_utils import launch
    launch(main)