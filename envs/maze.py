import torch
import numpy as np
import gym
import cv2
from gym import spaces
from tools.config import Configurable


def get_intersect(A, B, C, D):
    assert A.shape == B.shape == C.shape == D.shape
    assert A.shape[1] == 2

    A = A.permute(1, 0)
    B = B.permute(1, 0)
    C = C.permute(1, 0)
    D = D.permute(1, 0)

    det = (B[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (B[1] - A[1])

    no_intersect = (det.abs() < 1e-16)

    det = det.masked_fill(no_intersect, 1) # mask it as 1

    t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
    t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det

    no_intersect = torch.logical_or(no_intersect, (t1 > 1) | (t1 < 0) | (t2 > 1) | (t2 < 0))

    #xi = A[0] + t1 * (B[0] - A[0])
    #yi = A[1] + t1 * (B[1] - A[1])
    return no_intersect, t1



class Embedder:
    # https://github.com/yenchenlin/nerf-pytorch/blob/master/run_nerf_helpers.py
    def __init__(self, **kwargs):
        self.kwargs = kwargs

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d
            
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        
        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)
            
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d
                    
        self.embed_fns = embed_fns
        self.out_dim = out_dim
        
    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0):
    if i == -1:
        from torch import nn
        return nn.Identity(), 2
    
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 2,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [torch.sin, torch.cos],
    }
    
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim




class LargeMaze(Configurable):
    """Continuous maze environment."""
    SIZE = 12
    ACTION_SCALE=1.
    RESOLUTION = 512

    walls = torch.tensor(np.array(
        [
            [[-12.0, -12.0], [-12.0, 12.0]],
            [[-10.0, 8.0], [-10.0, 10.0]],
            [[-10.0, 0.0], [-10.0, 6.0]],
            [[-10.0, -4.0], [-10.0, -2.0]],
            [[-10.0, -10.0], [-10.0, -6.0]],
            [[-8.0, 4.0], [-8.0, 8.0]],
            [[-8.0, -4.0], [-8.0, 0.0]],
            [[-8.0, -8.0], [-8.0, -6.0]],
            [[-6.0, 8.0], [-6.0, 10.0]],
            [[-6.0, 4.0], [-6.0, 6.0]],
            [[-6.0, 0.0], [-6.0, 2.0]],
            [[-6.0, -6.0], [-6.0, -4.0]],
            [[-4.0, 2.0], [-4.0, 8.0]],
            [[-4.0, -2.0], [-4.0, 0.0]],
            [[-4.0, -10.0], [-4.0, -6.0]],
            [[-2.0, 8.0], [-2.0, 12.0]],
            [[-2.0, 2.0], [-2.0, 6.0]],
            [[-2.0, -4.0], [-2.0, -2.0]],
            [[0.0, 6.0], [0.0, 12.0]],
            [[0.0, 2.0], [0.0, 4.0]],
            [[0.0, -8.0], [0.0, -6.0]],
            [[2.0, 8.0], [2.0, 10.0]],
            [[2.0, -8.0], [2.0, 6.0]],
            [[4.0, 10.0], [4.0, 12.0]],
            [[4.0, 4.0], [4.0, 6.0]],
            [[4.0, 0.0], [4.0, 2.0]],
            [[4.0, -6.0], [4.0, -2.0]],
            [[4.0, -10.0], [4.0, -8.0]],
            [[6.0, 10.0], [6.0, 12.0]],
            [[6.0, 6.0], [6.0, 8.0]],
            [[6.0, 0.0], [6.0, 2.0]],
            [[6.0, -8.0], [6.0, -6.0]],
            [[8.0, 10.0], [8.0, 12.0]],
            [[8.0, 4.0], [8.0, 6.0]],
            [[8.0, -4.0], [8.0, 2.0]],
            [[8.0, -10.0], [8.0, -8.0]],
            [[10.0, 10.0], [10.0, 12.0]],
            [[10.0, 4.0], [10.0, 8.0]],
            [[10.0, -2.0], [10.0, 0.0]],
            [[12.0, -12.0], [12.0, 12.0]],
            [[-12.0, 12.0], [12.0, 12.0]],
            [[-12.0, 10.0], [-10.0, 10.0]],
            [[-8.0, 10.0], [-6.0, 10.0]],
            [[-4.0, 10.0], [-2.0, 10.0]],
            [[2.0, 10.0], [4.0, 10.0]],
            [[-8.0, 8.0], [-2.0, 8.0]],
            [[2.0, 8.0], [8.0, 8.0]],
            [[-10.0, 6.0], [-8.0, 6.0]],
            [[-6.0, 6.0], [-2.0, 6.0]],
            [[6.0, 6.0], [8.0, 6.0]],
            [[0.0, 4.0], [6.0, 4.0]],
            [[-10.0, 2.0], [-6.0, 2.0]],
            [[-2.0, 2.0], [0.0, 2.0]],
            [[8.0, 2.0], [10.0, 2.0]],
            [[-4.0, 0.0], [-2.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0]],
            [[6.0, 0.0], [8.0, 0.0]],
            [[-6.0, -2.0], [2.0, -2.0]],
            [[4.0, -2.0], [10.0, -2.0]],
            [[-12.0, -4.0], [-8.0, -4.0]],
            [[-4.0, -4.0], [-2.0, -4.0]],
            [[0.0, -4.0], [6.0, -4.0]],
            [[8.0, -4.0], [10.0, -4.0]],
            [[-8.0, -6.0], [-6.0, -6.0]],
            [[-2.0, -6.0], [0.0, -6.0]],
            [[6.0, -6.0], [10.0, -6.0]],
            [[-12.0, -8.0], [-6.0, -8.0]],
            [[-2.0, -8.0], [2.0, -8.0]],
            [[4.0, -8.0], [6.0, -8.0]],
            [[8.0, -8.0], [10.0, -8.0]],
            [[-10.0, -10.0], [-8.0, -10.0]],
            [[-4.0, -10.0], [4.0, -10.0]],
            [[-12.0, -12.0], [12.0, -12.0]],
        ]
    ))


    def __init__(self, cfg=None, batch_size=128, device='cuda:0', low_steps=200, reward=False, mode='batch', obs_dim=0) -> None:
        super().__init__()
        self.screen = None
        self.isopen = True
        self.device = device
        if mode != 'batch':
            batch_size = 1
        self.batch_size = batch_size



        self.action_space = spaces.Box(-1, 1, (2,))

        if obs_dim == 0:
            self.observation_space = spaces.Box(-12, 12, (2,))
        else:
            self.embedder, dim = get_embedder(obs_dim)
            self.observation_space = spaces.Box(-12, 12, (dim + 2,))
        self.obs_dim = obs_dim
        self.walls = self.walls.to(device).double()
        self.mode = mode
        self.render_wall()

    def get_obs(self):
        obs = self.pos.clone().float()
        if self.obs_dim == 0:
            return obs
        return torch.cat((obs * 0.01, self.embedder(obs)), -1)

    def step(self, action):
        if self.mode != 'batch':
            action = torch.tensor(action, device=self.device).double()[None, :]
        else:
            action = torch.tensor(action, device=self.device).double()
        assert self.pos.shape == action.shape

        new_pos = self.pos + action.clip(-1., 1.) * self.ACTION_SCALE

        collide = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)
        t = torch.ones(self.batch_size, device=self.device, dtype=self.pos.dtype)

        for wall in self.walls:
            left = wall[0][None, :].expand(self.batch_size, -1)
            right = wall[1][None, :].expand(self.batch_size, -1)
            no_collide, new_t = get_intersect(self.pos, new_pos, left, right)
            new_collide = torch.logical_not(no_collide)

            if t is None:
                t = new_t
                collide = new_collide
            else:
                t = torch.where(new_collide, torch.min(t, new_t), t)
                collide = torch.logical_or(collide, new_collide)

        intersection = self.pos + torch.relu(t[:, None] - 1e-5) * (new_pos - self.pos)
        new_pos = torch.where(collide[:, None], intersection, new_pos)

        self.pos = new_pos

        reward = torch.zeros(self.batch_size, device=self.device) + self.get_reward()
        if self.mode == 'batch':
            return self.get_obs(), reward.float(), False, {}
        else:
            o, r = self.get_obs().detach().cpu().numpy()[0], reward.detach().cpu().numpy().reshape(-1)[0]
            return o, r, False, {}


    def get_reward(self):
        if self._cfg.reward:
            return self.pos.sum(axis=-1)
        else:
            return 0
        
    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        self.pos = torch.zeros(self.batch_size, 2, device=self.device, dtype=torch.float64)

        if self.mode == 'batch':
            return self.get_obs()
        else:
            return self.get_obs().detach().cpu().numpy()[0]


    def pos2pixel(self, x):
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = (x + self.SIZE) / (self.SIZE * 2) * 0.8 + 0.1
        return (x* self.RESOLUTION).astype(np.int32)

    def render_wall(self):
        import cv2
        resolution = self.RESOLUTION
        self.screen = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for wall in self.walls:
            left = wall[0].detach().cpu().numpy()
            right = wall[1].detach().cpu().numpy()
            #left = ((left + self.SIZE) / (self.SIZE * 2) * resolution).astype(np.int32)
            # left = self.pos2pixel((left + self.SIZE) / (self.SIZE * 2))
            # right = self.pos2pixel((right + self.SIZE) / (self.SIZE * 2))
            # right = ((right + self.SIZE) / (self.SIZE * 2) * resolution).astype(np.int32)
            left = self.pos2pixel(left)
            right = self.pos2pixel(right)
            cv2.line(self.screen, tuple(left), tuple(right), (255, 255, 255), 1)
        return self.screen

    def render(self, mode):
        assert mode == 'rgb_array'

        img = self.screen.copy()
        # pos = ((self.pos.detach().cpu().numpy() + self.SIZE) / (self.SIZE * 2) * 512).astype(np.int32)
        pos = self.pos2pixel(self.pos)
        for i in range(pos.shape[0]):
            cv2.circle(img, tuple(pos[i]), 3, (0, 255, 0), 1)
        return img

    def _render_traj_rgb(self, traj, z=None, **kwargs):
        # assert z is None
        from tools.utils import plt_save_fig_array
        import matplotlib.pyplot as plt

        if isinstance(traj, dict):
            obs = traj['next_obs']
        else:
            obs = traj.get_tensor('next_obs')

        obs = obs[..., :2]
        if self.obs_dim > 0:
            obs = obs / 0.01

        plt.clf()
        img = self.screen.copy()/255.
        #obs = ((obs  + self.SIZE)/ (self.SIZE * 2)).detach().cpu().numpy()
        #obs = obs / self.SIZE / 2 + 0.5
        #obs = obs.detach().cpu().numpy() * self.RESOLUTION
        obs = self.pos2pixel(obs)
        return {
            'state': obs,
            'background': {
                'image':  img,
                'xlim': [0, self.RESOLUTION],
                'ylim': [0, self.RESOLUTION],
            },
        }


class SmallMaze(LargeMaze):
    SIZE = 5

    walls = torch.tensor(np.array(
        [
            [[-5, -5], [-5., 5.0]],
            [[-5, 5], [5., 5.0]],
            [[5, 5], [5., -5.0]],
            [[5, -5], [-5., -5.0]],

            [[-1., 1.], [1., 1.]],
            [[-1., 1.], [-1., 5.]],
            [[1., 1.], [1., 5.]],

            [[-1., -1.], [1., -1.]],
            [[-1., -1.], [-1., -5.]],
            [[1., -1.], [1., -5.]],

            [[-1.5, -1.], [-1.5, 1.]],
            [[-1.5, -1.], [-5., -1.]],
            [[-1.5, 1.], [-5., 1.]],

            [[1.5, -1.], [1.5, 1.]],
            [[1.5, -1.], [5., -1.]],
            [[1.5, 1.], [5., 1.]],

        ]))

    def __init__(self, cfg=None, low_steps=20) -> None:
        super().__init__(cfg)
        self.reset()


class MediumMaze(LargeMaze):
    # from pacman.maze import extract_mazewall
    SIZE = 7
    walls = torch.tensor(np.array(
[[[-7, -1], [-5, -1]], [[-7, 5], [-5, 5]], [[-5, -5], [-3, -5]], [[-5, -3], [-3, -3]], [[-5, -1], [-3, -1]], [[-5, 1], [-3, 1]], [[-5, 1], [-5, 3]], [[-5, 3], [-3, 3]], [[-5, 5], [-3, 5]], [[-3, -5], [-1, -5]], [[-3, -3], [-1, -3]], [[-3, -1], [-1, -1]], [[-3, 1], [-1, 1]], [[-3, 3], [-1, 3]], [[-1, -7], [-1, -5]], [[-1, -3], [1, -3]], [[-1, 1], [1, 1]], [[-1, 3], [-1, 5]], [[1, -5], [3, -5]], [[1, -5], [1, -3]], [[1, -3], [1, -1]], [[1, -1], [1, 1]], [[1, 3], [1, 5]], [[1, 5], [1, 7]], [[3, -5], [3, -3]], [[3, -3], [5, -3]], [[3, -3], [3, -1]], [[3, -1], [3, 1]], [[3, 1], [5, 1]], [[3, 1], [3, 3]], [[3, 3], [3, 5]], [[5, -5], [7, -5]], [[5, -1], [7, -1]], [[5, 1], [5, 3]], [[5, 3], [7, 3]], [[5, 5], [5, 7]], [[-7, -7], [-7, 7]], [[-7, 7], [7, 7]], [[7, 7], [7, -7]], [[7, -7], [-7, -7]]]
        ))

    def __init__(self, cfg=None, low_steps=40) -> None:
        super().__init__(cfg)
        self.reset()

    
if __name__ == '__main__':
    env = SmallMaze()
    import matplotlib.pyplot as plt
    plt.imshow(env.render('rgb_array'))
    plt.savefig('xx.png')