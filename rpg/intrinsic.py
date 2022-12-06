# intrinsic motivation
import torch

class IntrinsicMotivation:
    def __init__(self, *args) -> None:
        self.args = args

    def __call__(self, rollout):
        outs = []
        reward = rollout['reward']

        for i in self.args:
            intrinsic = i.intrinsic_reward(rollout)
            assert reward.shape == intrinsic.shape
            outs.append(intrinsic)
        return reward, torch.cat(outs, dim=-1)