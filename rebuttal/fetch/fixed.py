import numpy as np
from ..env_builder import register

def compute_action(env, goal, grasp, k=10.):
    return np.append(k * (goal - env.get_gripper_pose()), grasp).clip(-1, 1)

def moveto(env, goal, gripper, k=10, eps=0.02):
    imgs = []
    idx = 0
    while np.linalg.norm(env.get_gripper_pose() - goal) > eps and idx < 100:
        env.step(compute_action(env, goal, gripper, k)+(np.random.random((4,))*2 - 1)*0.01)
        idx += 1
    return imgs

def grasp(env, obj_id, release=True):
    #env.get_gripper_pose()
    pose = env.get_gripper_pose()
    obj_pos = env.get_block_pose(obj_id)
    pose[:2] = obj_pos[:2]
    pose[2] = max(pose[2], 0.1)
    imgs = []
    imgs += moveto(env, pose, 1)
    pose = obj_pos[:]
    pose[2] += 0.02
    imgs += moveto(env, pose, 1, eps=0.002)
    for i in range(2):
        env.step([0, 0, 0, -1])
        imgs.append(env.render(mode='rgb_array'))
    goal = env.goal[obj_id*3:obj_id*3+3]

    pose[2] = max(goal[2] + 0.02, 0.1)
    imgs += moveto(env, pose, -1)

    pose[:2] = goal[:2]
    imgs += moveto(env, pose, -1)
    if release:
        pose[2] = goal[2] + 0.01
        imgs += moveto(env, pose, -1, eps=0.002)

        pose[2] += 0.1
        imgs += moveto(env, pose, 1)
    return imgs



from .sequential import SequentialStack
class Fixed(SequentialStack):
    def __init__(self, cfg=None):
        SequentialStack.__init__(self, cfg)

        self.env.seed(1)
        self.env.reset()
        grasp(self.env, 0, False)

        self.state = self.env.get_state()

    def reset(self):
        super(Fixed, self).reset()
        self.env.set_state(self.state)
        self.steps = self.substeps
        self.obj_id = 1
        return self._get_obs()

register(Fixed)
