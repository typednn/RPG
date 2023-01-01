import torch
from tools import Configurable
from tools.utils import totensor, tonumpy
from .buffer import ReplayBuffer
from collections import deque
import numpy as np
from tools.utils import RunningMeanStd
from tools.optim import OptimModule


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


class ExplorationBonus(OptimModule):
    name = 'bonus'
    def __init__(self, module, buffer: ReplayBuffer, enc_s,
                 cfg=None, buffer_size=None, update_step=1, update_freq=1, batch_size=512,
                 obs_mode='state',
                 normalizer=None,

                 as_reward=True,
                 training_on_rollout=False,
                 scale=0.,

                 include_latent=False,
    ) -> None:

        super().__init__(module)
        if buffer_size is None:
            buffer_size = buffer.capacity

        self.step = 0
        self.buffer = deque(maxlen=buffer_size)
        self.bufferz = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.enc_s = enc_s

        if normalizer is None or normalizer == 'none':
            self.normalizer = None
        elif normalizer == 'ema':
            self.normalizer = RunningMeanStd(last_dim=True)
        elif isinstance(normalizer, int):
            self.normalizer = ScalarNormalizer(normalizer)
        else:
            raise NotImplementedError

            
        self.as_reward = as_reward
        self.training_on_rollout = training_on_rollout
        self.obs_mode = obs_mode
        self.update_step = update_step
        self.update_freq = update_freq
        
        self.scale = scale

        self.include_latent = include_latent
        if include_latent:
            assert self.obs_mode != 'state'
            assert not self.training_on_rollout

    def add_data(self, data, prevz):
        if not self.training_on_rollout:
            with torch.no_grad():
                for i, z in zip(data, prevz):
                    self.buffer.append(i)
                    self.bufferz.append(z)

            self.step += 1
            if self.step % self.update_freq == 0 and len(self.buffer) > self.batch_size:
                for _ in range(self.update_step):
                    bonus = self.update(*self.sample_data())
                    if self.normalizer is not None:
                        self.normalizer.update(bonus)

    def process_obs(self, obs):
        if self.obs_mode == 'state':
            return self.enc_s(obs)
        else:
            return obs

    def sample_data(self):
        #pass
        assert not self.training_on_rollout
        index = np.random.choice(len(self.buffer), self.batch_size)
        obs = totensor([self.buffer[i] for i in index], device='cuda:0')
        if self.include_latent:
            latent = totensor([self.bufferz[i] for i in index], device='cuda:0', dtype=None)
        else:
            latent = None
        obs = self.process_obs(obs)
        return obs, latent

    def update(self, inp, latent) -> torch.Tensor: 
        # the input is the same with the compute_bonus
        raise NotImplementedError

    def compute_bonus(self, inp, latent) -> torch.Tensor:
        # when obs_mode is state, obs is the state
        # otherwise, it is the obs
        raise NotImplementedError

    def compute_bonus_by_obs(self, obs, latent):
        obs = self.process_obs(obs)
        return self.compute_bonus(obs, latent)
        
    def visualize_transition(self, transition):
        attrs = transition.get('_attrs', {})
        if 'r' not in attrs:
            from tools.utils import tonumpy
            attrs['r'] = tonumpy(transition['r'])[..., 0] # just record one value ..

        if self.obs_mode == 'state':
            attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_state'], transition['z']))
        else:
            attrs['bonus'] = tonumpy(self.compute_bonus(transition['next_obs'], transition['z']))
        transition['_attrs'] = attrs


    # for training on state
    def update_by_rollout(self, rollout):
        #raise NotImplementedError
        if self.training_on_rollout:
            assert self.obs_mode == 'state'
            bonus = self.update(rollout['state'][1:].detach())
            if self.normalizer is not None:
                self.normalizer.update(bonus)

    def intrinsic_reward(self, rollout, latent):
        if not self.as_reward:
            assert self.obs_mode == 'state'
            bonus = self.compute_bonus(rollout['state'][1:])
            if self.normalizer is not None:
                bonus = self.normalizer.normalize(bonus)
            return self.name, bonus * self.scale
        else:
            # rollout is the obs
            bonus = self.compute_bonus_by_obs(rollout, latent)
            if self.normalizer is not None:
                bonus = self.normalizer.normalize(bonus)
            return bonus * self.scale
            