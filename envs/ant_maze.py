# ant environment for exploration
import numpy as np
import torch
import gym
from envs.pacman.antman import AntManEnv
from .maze import get_embedder


class AntMaze(gym.Env):
    def __init__(self, obs_dim=8, reward=False) -> None:
        super().__init__()
        assert not reward
        self.reward = reward

        self.ant_env = AntManEnv(reset_maze=False)

        self.obs_dim = obs_dim
        if self.obs_dim > 0:
            self.embedder, _ = get_embedder(obs_dim)

        self.grid_size = self.ant_env.ant_env.MAZE_SIZE_SCALING
        obs = self.reset()
        self.init_local_obs = self.low_obs.copy()

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)
        self.action_space = self.ant_env.action_space

        self.action_scale = self.action_space.high[0]
        self.action_space = gym.spaces.Box(-1, 1, shape=self.action_space.shape, dtype=np.float32)

        self.device = 'cuda:0'

        
    def decorate_obs(self, obs):
        if self.obs_dim > 0:
            pos = obs[:2] / self.grid_size / 4
            obs = np.concatenate(
                [pos/100., obs[2:]/10., self.embedder(
                    torch.tensor(pos)).detach().cpu().numpy()])
        return obs

    def get_obs(self):
        return self.decorate_obs(self.low_obs.copy())

    def reset(self):
        self.low_obs = self.ant_env.reset(player=[3, 3])
        return self.get_obs()

    def step(self, action):
        # add clip ..
        self.low_obs, _, _, _ = self.ant_env.step(action.clip(-1., 1.) * self.action_scale)
        reward = 0.
        return self.get_obs(), reward, False, {}

    def render(self, mode='rgb_array'):
        return self.ant_env.render(mode=mode)

    def get_obs_from_traj(self, traj):
        if isinstance(traj, dict):
            obs = traj['next_obs']
        else:
            obs = traj.get_tensor('next_obs')
        obs = obs[..., :2]
        if self.obs_dim > 0:
            obs = obs * 100 * self.grid_size * 4
        return obs

    def _render_traj_rgb(self, traj, z=None, occ_val=False, history=None, **kwargs):
        obs = self.get_obs_from_traj(traj)

        if occ_val >= 0:
            occupancy = self.counter(obs) 
            if history is not None:
                occupancy += history['occ']
        else:
            occupancy = None

        obs = obs.detach().cpu().numpy()
        output = {
            'state': obs,
            'background': {
                'image':  None,
                'xlim': [0, self.grid_size * 4],
                'ylim': [0, self.grid_size * 4],
            },
            'image': {'occupancy': occupancy / occupancy.max()},
            'history': {'occ': occupancy},
            'metric': {'occ': (occupancy > occ_val).mean()},
        }

        return output

    def build_anchor(self):
        x = torch.arange(0., 4, device=self.device) + 0.5
        y = torch.arange(0., 4, device=self.device) + 0.5
        x, y = torch.meshgrid(x, y, indexing='ij')
        return torch.stack([y, x], dim=-1).cuda()

    def counter(self, obs):
        anchor = self.build_anchor()
        obs = obs.reshape(-1, obs.shape[-1])/self.grid_size
        #print(obs.shape, anchor.shape)
        reached = torch.abs(obs[None, None, :, :] - anchor[:, :, None, :])
        reached = torch.logical_and(reached[..., 0] < 0.5, reached[..., 1] < 0.5)
        return reached.sum(axis=-1).float().detach().cpu().numpy()[::-1]

    @property
    def anchor_state(self):
        init_obs = self.init_local_obs
        #for x in range(4):
        #    for y in range(4):
        anchor_obs = self.build_anchor().detach().cpu().numpy()
        
        outs = []
        coords = []
        for i in anchor_obs.reshape(-1, 2):
            # print(i * self.grid_size, init_obs[:2])
            coords.append(i * self.grid_size)
            outs.append(
                self.decorate_obs(np.concatenate([i * self.grid_size, init_obs[2:]]))
            )
            # print(outs[-1])
        return {
            'state': np.stack(outs),
            'coord': np.stack(coords),
        }