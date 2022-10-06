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

class ContinuousMaze(Configurable):
    """Continuous maze environment."""

    action_space = spaces.Box(-1, 1, (2,))
    observation_space = spaces.Box(-12, 12, (2,))

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


    def __init__(self, cfg=None, batch_size=128, device='cuda:0', low_steps=200) -> None:
        super().__init__()
        self.screen = None
        self.isopen = True
        self.device = device
        self.batch_size = batch_size
        self.walls = self.walls.to(device)
        self.render_wall()


    def step(self, action):
        assert self.pos.shape == action.shape

        new_pos = self.pos + action * 0.3

        collide, t = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device), torch.ones(self.batch_size, device=self.device)

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

        intersection = self.pos + (t[:, None] - 1e-5) * (new_pos - self.pos)
        new_pos = torch.where(collide[:, None], intersection, new_pos)

        self.pos = new_pos
        return self.pos.clone(), torch.zeros(self.batch_size, device=self.device), False, {}
        
    def reset(self, batch_size=None):
        if batch_size is not None:
            self.batch_size = batch_size

        self.pos = torch.zeros(self.batch_size, 2, device=self.device)
        return self.pos.clone()

    def render_wall(self):
        import cv2
        resolution = 512
        self.screen = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for wall in self.walls:
            left = wall[0].detach().cpu().numpy()
            right = wall[1].detach().cpu().numpy()
            left = ((left + 12) / 24 * resolution).astype(np.int32)
            right = ((right + 12) / 24 * resolution).astype(np.int32)
            cv2.line(self.screen, tuple(left), tuple(right), (255, 255, 255), 1)
        return self.screen

    def render(self, mode):
        assert mode == 'rgb_array'

        img = self.screen.copy()
        pos = ((self.pos.detach().cpu().numpy() + 12) / 24 * 512).astype(np.int32)
        for i in range(pos.shape[0]):
            cv2.circle(img, tuple(pos[i]), 3, (0, 255, 0), 1)
        return img

        
if __name__ == '__main__':
    env = ContinuousMaze()
    import cv2
    cv2.imwrite('tmp.png', env.screen)