from tools.config import Configurable
import einops
import re
import numpy as np
import torch
from .worldmodel import HiddenDynamicNet

def linear_schedule(schdl, step):
    """
    Outputs values following a linear decay schedule.
    Adapted from https://github.com/facebookresearch/drqv2
    """
    try:
        return float(schdl)
    except ValueError:
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
    raise NotImplementedError(schdl)


class CEM(Configurable):
    def __init__(self, trainer, horizon, cfg=None, seed_steps=1000,
                num_samples=512,
                iterations=6,
                mixture_coef=0.05,
                min_std = 0.05,
                temperature= 0.5
                momentum= 0.1,
                num_elites=64,
        ) -> None:
        super().__init__()
        from .soft_rpg import Trainer
        trainer: Trainer = trainer
        self.cfg = cfg
        self.horizon = horizon
        self.worldmodel: HiddenDynamicNet = trainer.dynamics_net
        self.horizon_schedule = f'linear(1, {horizon}, 25000)'


    def plan(self, obs, timesteps, z, eval_mode=False, step=None):
        # copied from tdmpc
        """
        Plan next action using TD-MPC inference.
        obs: raw input observation.
        eval_mode: uniform sampling and action noise is disabled during evaluation.
        step: current time step. determines e.g. planning horizon.
        t0: whether current step is the first step of an episode.
        """
        # Seed steps
        if step < self.cfg.seed_steps and not eval_mode:
            return torch.empty(self.cfg.action_dim, dtype=torch.float32, device=self.device).uniform_(-1, 1)

        # Sample policy trajectories
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        horizon = int(min(self.horizon, linear_schedule(self.cfg.horizon_schedule, step)))
        num_pi_trajs = int(self.cfg.mixture_coef * self.cfg.num_samples)

        batch_size = obs.shape[0]
        if num_pi_trajs > 0:
            _obs, _z, _t = [einops.repeat(x, 'b ... -> (n b) ...', n=num_pi_trajs) for x in [obs, z, timesteps]]
            data = self.worldmodel.inference(_obs, _z, _t, horizon)
            pi_actions = einops.rearrange(data['a'], '(n b) ... -> n b ...', n=num_pi_trajs) # ((b n) n)

        action_dim = pi_actions.shape[-1]

        # Initialize state and parameters
        #z = self.model.h(obs).repeat(self.cfg.num_samples+num_pi_trajs, 1)
        device = pi_actions.device
        mean = torch.zeros(horizon, batch_size, action_dim, device=device)
        std = 2*torch.ones(horizon, batch_size, action_dim, device=device)

        t0 = (timesteps == 0)
        if hasattr(self, '_prev_mean') and self._prev_mean is not None and t0.any():
            mean[t0, :-1] = self._prev_mean[t0, 1:]

        # Iterate CEM
        obs, z, timesteps = [einops.repeat(x, 'b ... -> (n b) ...', n=num_pi_trajs + self.cfg.num_samples) 
                       for x in [obs, z, timesteps]]

        for i in range(self.cfg.iterations):
            actions = torch.clamp(mean.unsqueeze(1) + std.unsqueeze(1) * \
                torch.randn(horizon, self.cfg.num_samples, batch_size, action_dim, device=std.device), -1, 1)

            if num_pi_trajs > 0:
                actions = torch.cat([actions, pi_actions], dim=1) # T, n, b, action_dim

            # Compute elite actions
            #value = self.estimate_value(z, actions, horizon).nan_to_num_(0)

            # (b n) 1
            value = self.worldmodel.inference(obs, z, timesteps, horizon, a_seq=actions)['value']
            value = einops.rearrange(value, '(n b) ... -> n b ...', b=batch_size)

            elite_idxs = torch.topk(value[..., 0], self.cfg.num_elites, dim=0).indices # K, b
            elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

            # Update parameters
            max_value = elite_value.max(0)[0]
            score = torch.exp(self.cfg.temperature*(elite_value - max_value))
            score /= score.sum(0)
            _mean = torch.sum(score.unsqueeze(0) * elite_actions, dim=1) / (score.sum(0) + 1e-9)
            _std = torch.sqrt(torch.sum(score.unsqueeze(0) * (elite_actions - _mean.unsqueeze(1)) ** 2, dim=1) / (score.sum(0) + 1e-9))
            _std = _std.clamp_(self.std, 2)
            mean, std = self.cfg.momentum * mean + (1 - self.cfg.momentum) * _mean, _std

        # Outputs
        score = score.squeeze(1).cpu().numpy()
        actions = elite_actions[:, np.random.choice(np.arange(score.shape[0]), p=score)]
        self._prev_mean = mean
        mean, std = actions[0], _std[0]
        a = mean
        if not eval_mode:
            a += std * torch.randn(self.cfg.action_dim, device=std.device)
        return a
