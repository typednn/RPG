import torch
from tools import Configurable
from tools.utils import totensor, tonumpy
from .buffer import ReplayBuffer
from collections import deque
import numpy as np
from tools.utils import RunningMeanStd

"""
The intrinsic reward:

Data-source:

Data-source and When to do the update:
    # - onpolicy: trained with the off-policy data sampled from the replay buffer 
    # - off-policy: trained with the recent data, (or only maintain a buffer of fixed size)
    no matter what or not, we just need to maintain a buffer of fixed size to mimic the two

    - buffer size
    - parameters:
        - update frequency (=1, per step, or epoch length per epoch)
        - update steps / epochs

---
Reward mode:
    - if the input is the encoding state: we can directly use it as the intrinsic reward ..
    - otherwise, use it to modify the original reward .. 

Observation:
    - Use state or observation as the input?

Normalization:
    - normalization with EMA.. this does not seem to work well as the loss decays exponentially
    - another way is to maintain a loss buffer and normalize them correspondingly

"""

class ScalarNormalizer:
    # maintain a deque of fixed size and update the parameters with the recent data
    def __init__(self, size=10000):
        self.size = size
        self.data = deque()
        self.sum = 0.
        self.sumsq = 1.
        self.step = 0

    def update(self, data):
        sum = float(data.sum())
        sumsq = float((data**2).sum())
        self.data.append([sum, sumsq])
        self.sum += sum
        self.sumsq += sumsq
        if len(self.data)> self.size:
            sum, sumsq = self.data.popleft()
            self.sum -= sum
            self.sumsq -= sumsq

    @property
    def mean(self):
        return self.sum / len(self.data)

    @property
    def std(self):
        return np.sqrt(self.sumsq / len(self.data) - self.mean**2 + 1e-16)

    def normalize(self, data):
        return (data - self.mean) / self.std


class ExplorationBonus(Configurable):
    name = 'bonus'
    def __init__(self, buffer: ReplayBuffer, enc_s,
                 cfg=None, buffer_size=None, update_step=1, update_freq=1, batch_size=512,
                 obs_mode='state', normalizer=None, as_reward=False) -> None:
        super().__init__()
        if buffer_size is None:
            buffer_size = buffer.capacity

        self.step = 0
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.enc_s = enc_s

        if normalizer is None:
            self.normalizer = None
        elif normalizer == 'ema':
            self.normalizer = RunningMeanStd(last_dim=True)
        elif isinstance(normalizer, int):
            self.normalizer = ScalarNormalizer(normalizer)
        else:
            raise NotImplementedError("normalizer: {}".format(normalizer))

            
        self.as_reward = as_reward
        self.obs_mode = obs_mode
        self.update_step = update_step
        self.update_freq = update_freq

    def add_data(self, data):
        self.buffer.append(data)
        self.step += 1
        if self.step % self.update_freq == 0:
            for _ in range(self.update_step):
                bonus = totensor(self.update(self.sample_data()))
                if self.normalizer is not None:
                    self.normalizer.update(bonus)

    def sample_data(self):
        #pass
        index = np.random.choice(len(self.buffer), self.batch_size)
        obs = totensor([self.buffer[i] for i in index])
        if self.obs_mode == 'state':
            obs = self.enc_s(obs)
        return obs

    def update(self, obs) -> torch.Tensor: 
        raise NotImplementedError

    def compute_bonus(self, obs) -> torch.Tensor:
        raise NotImplementedError
        
    def intrinsic_reward(self, rollout):
        if not self.as_reward:
            assert self.obs_mode == 'state'
            bonus = self.compute_bonus(rollout['state'][1:])
            if self.normalizer is not None:
                bonus = self.normalizer.normalize(bonus)
            return self.name, bonus
        else:
            return None, None
            
    def visualize_transition(self, transition):
        attrs = transition.get('_attrs', {})
        if 'r' not in attrs:
            from tools.utils import tonumpy
            attrs['r'] = tonumpy(transition['r'])[..., 0] # just record one value ..

        if self.obs_mode == 'state':
            attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_state']))
        else:
            attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_obs']))
        transition['_attrs'] = attrs