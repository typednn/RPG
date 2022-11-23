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

    no_intersect = (det.abs() < 1e-10)

    det = det.masked_fill(no_intersect, 1) # mask it as 1

    t1 = ((C[0] - A[0]) * (C[1] - D[1]) - (C[0] - D[0]) * (C[1] - A[1])) / det
    t2 = ((B[0] - A[0]) * (C[1] - A[1]) - (C[0] - A[0]) * (B[1] - A[1])) / det

    no_intersect = torch.logical_or(no_intersect, (t1 > 1) | (t1 < 0) | (t2 > 1) | (t2 < 0))

    #xi = A[0] + t1 * (B[0] - A[0])
    #yi = A[1] + t1 * (B[1] - A[1])
    return no_intersect, t1




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


    def __init__(self, cfg=None, batch_size=128, device='cuda:0', low_steps=200, reward=False, mode='batch') -> None:
        super().__init__()
        self.screen = None
        self.isopen = True
        self.device = device
        if mode != 'batch':
            batch_size = 1
        self.batch_size = batch_size



        self.action_space = spaces.Box(-1, 1, (2,))
        self.observation_space = spaces.Box(-12, 12, (2,))
        self.walls = self.walls.to(device).float()
        self.mode = mode
        self.render_wall()


    def step(self, action):
        if self.mode != 'batch':
            action = torch.tensor(action, device=self.device).float()[None, :]
        else:
            action = torch.tensor(action, device=self.device).float()
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
            return self.pos.clone(), reward, False, {}
        else:
            o, r = self.pos.clone().detach().cpu().numpy()[0], reward.detach().cpu().numpy().reshape(-1)[0]
            return o, r, False, {}


    def get_reward(self):
        if self._cfg.reward:
            return self.pos.sum(axis=-1)
        else:
            return 0
        
    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        self.pos = torch.zeros(self.batch_size, 2, device=self.device, dtype=torch.float32)

        if self.mode == 'batch':
            return self.pos.clone()
        else:
            return self.pos.clone().detach().cpu().numpy()[0]


    def render_wall(self):
        import cv2
        resolution = self.RESOLUTION
        self.screen = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for wall in self.walls:
            left = wall[0].detach().cpu().numpy()
            right = wall[1].detach().cpu().numpy()
            left = ((left + self.SIZE) / (self.SIZE * 2) * resolution).astype(np.int32)
            right = ((right + self.SIZE) / (self.SIZE * 2) * resolution).astype(np.int32)
            cv2.line(self.screen, tuple(left), tuple(right), (255, 255, 255), 1)
        return self.screen

    def render(self, mode):
        assert mode == 'rgb_array'

        img = self.screen.copy()
        pos = ((self.pos.detach().cpu().numpy() + self.SIZE) / (self.SIZE * 2) * 512).astype(np.int32)
        for i in range(pos.shape[0]):
            cv2.circle(img, tuple(pos[i]), 3, (0, 255, 0), 1)
        return img

    def _render_traj_rgb(self, traj, z=None, **kwargs):
        # assert z is None
        from tools.utils import plt_save_fig_array
        import matplotlib.pyplot as plt

        if isinstance(traj, dict):
            obs = traj['obs']
        else:
            raise NotImplementedError

        plt.clf()
        img = self.screen.copy()/255.
        #obs = ((obs  + self.SIZE)/ (self.SIZE * 2)).detach().cpu().numpy()
        obs = obs / self.SIZE / 2 + 0.5
        obs = obs.detach().cpu().numpy() * self.RESOLUTION


        from solver.draw_utils import plot_colored_embedding
        plt.clf()
        if z is not None and (z.max() < 100 or z.dtype != torch.int64):
            plt.imshow(np.uint8(img[...,::-1]*255))
            plot_colored_embedding(z, obs[1:, :, :2], s=3)
        else:
            plt.imshow(img[...,::-1])
            obs = obs.reshape(-1, 2)
            plt.scatter(obs[:, 0], obs[:, 1], s=3)

        plt.xlim([0, self.RESOLUTION])
        plt.ylim([0, self.RESOLUTION])
        plt.axis('off')
        plt.tight_layout(rect=[0.,0.,1., 1.])
        img2 = plt_save_fig_array()[:, :, :3]
        return img2


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




from rpg.common_hooks import as_hook, HookBase
                    

@as_hook
class plot_maze_env_rnd(HookBase):
    def __init__(self, resolution=64) -> None:
        super().__init__()
        self.resolution = resolution

    def init(self, trainer):
        # patch
        self.trainer = trainer
        self.env = trainer.env.goal_env
        self.old_render = trainer.env.goal_env._render_traj_rgb
        trainer.env.goal_env._render_traj_rgb = self._render_traj_rgb

    def  _render_traj_rgb(self, obs, z=None, **kwargs):
        import matplotlib.pyplot as plt
        from tools.utils import plt_save_fig_array
        plt.clf()
        plt.imshow(self.render_rnd())
        plt.tight_layout(rect=[0., 0., 1., 1.])
        plt.axis('off')
        img2 = plt_save_fig_array()[:, :, :3]
        img = self.old_render(obs, z)
        return np.concatenate((img, img2), axis=1)


    def render_rnd(self):
        import numpy as np
        from rpg.rnd import RNDOptim

        env: LargeMaze = self.env
        rnd: RNDOptim = self.trainer.rnd

        x, y = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        obs = np.stack([x, y], axis=2).reshape(-1, 2) / self.resolution
        obs = obs  * env.SIZE * 2 - env.SIZE

        out = rnd.compute_loss(obs, None, None, reduce=False)
        out = out - out.min()
        out = out / (out.max() + 1e-9)

        images = out.reshape(self.resolution, self.resolution).detach().cpu().numpy()
        return images

    
if __name__ == '__main__':
    env = SmallMaze()
    import matplotlib.pyplot as plt
    plt.imshow(env.render('rgb_array'))
    plt.savefig('xx.png')