# https://github.com/hzaskywalker/hrl/blob/main/hrl/envs/block/multiblock.py

import numpy as np
import sapien.core as sapien
import gym
from .sapien_sim import SimulatorBase, Pose, load_state_vector, state_vector
from tools.config import Configurable, as_builder
from typing import Optional

import numpy as np
from gym import Env, spaces

from .sapien_sim import SimulatorBase, Pose
from .sapien_utils import add_link, identity, x2y




def sample_blocks(n, block_size, world_size, previous=()):
    objects = list(previous)
    for i in range(n):
        not_found = True
        while not_found:
            not_found = False
            xy = (np.random.random((2,)) * 2 - 1) * world_size
            for j in objects:
                if np.abs(j - xy).min() < block_size:
                    not_found = True
                    break
        objects.append(xy)
    return objects[len(previous):]


COLORS = [
    [0.25, 0.25, 0.75],
    [0.25, 0.75, 0.25],
    [0.25, 0.75, 0.75],
]



def set_actor_xy(actor, xy):
    p = actor.get_pose()
    new_p = p.p
    new_p[:2] = np.array(xy)
    actor.set_pose(Pose(new_p, p.q))


class BlockEnv(gym.Env, SimulatorBase):
    def __init__(
        self, dt=0.01,
        frameskip=8,
        gravity=(0, 0, 0),
        random_blocks=False,
        random_goals=False,
        n_block=3,
    ):

        
        self.n_block = n_block
        self.blocks = []
        self.goal_vis = []

        self.block_size = 0.2
        self.wall_width = 0.1
        self.world_size = 2 - self.block_size
        self.success_threshold = self.block_size

        self.random_blocks = random_blocks
        self.random_goals = random_goals

        super(BlockEnv, self).__init__(dt, frameskip, gravity)

        
        self.agent = self.add_articulation(
            -1, 0, self.block_size, (0.75, 0.25, 0.25),
            self.ranges, "pusher", friction=0, damping=0, material=self.material)

        obs = self.reset()
        self.observation_space = spaces.Box(low=-4, high=4, shape=(len(obs),))
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))


    def get_agent_pos(self):
        return self.agent.get_qpos()

    def get_agent_vel(self):
        return self.agent.get_qvel()

    def _get_obs(self):
        pos, vel, diff = [], [], []
        for b, g in zip(self.blocks, self.goal_vis):
            pos.append(b.get_qpos())
            vel.append(b.get_qvel())
            diff.append(g.get_pose().p[:2] - b.get_qpos())
        return np.concatenate(pos + vel + diff)


    def step(self, action: np.ndarray):
        action = np.asarray(action)

        for i in range(self.frameskip):
            self.agent.set_qvel(action.clip(-1, 1) * 4)
            self._scene.step()

        r = self.compute_reward()
        # d = np.linalg.norm(self.block.get_qpos() - self.goal)
        return self._get_obs(), self.compute_reward(), False, {}

    def reset(self):
        SimulatorBase.reset(self)

        
        if self.random_blocks:
            xys = sample_blocks(self.n_block, self.block_size, self.world_size, previous=())
            for b, xy in zip(self.blocks, xys):
                b.set_qpos(xy)

        if self.random_goals:
            goals = sample_blocks(self.n_block, self.block_size, self.world_size, previous=())
            for g, xy in zip(self.goal_vis, goals):
                set_actor_xy(g, xy)


        return self._get_obs()

    def render(self, mode='human'):
        return SimulatorBase.render(self, mode)

    

    def get_goals(self):
        goals = []
        for g in self.goal_vis:
            goals.append(g.get_pose().p[:2])
        return np.array(goals)

    def compute_reward(self):
        r = 0
        self.success = 0
        for b, g in zip(self.blocks, self.goal_vis):
            dist = np.linalg.norm(b.get_qpos() - g.get_pose().p[:2])
            self.success += dist < self.success_threshold
            r += dist ** 2
        return - r ** 0.5


    def build_scene(self):
        np.random.seed(0)

        wall_color = (0.3, 0.7, 0.3)

        wall_width = self.wall_width
        world_width = self.world_size + self.block_size + self.wall_width
        self.add_box(-world_width, 0, (wall_width, world_width - wall_width, 0.5), wall_color, 'wall1', True, False)
        self.add_box(+world_width, 0, (wall_width, world_width - wall_width, 0.5), wall_color, 'wall2', True, False)
        self.add_box(0, +world_width, (world_width + wall_width, wall_width, 0.5), wall_color, 'wall3', True, False)
        self.add_box(0, -world_width, (world_width + wall_width, wall_width, 0.5), wall_color, 'wall4', True, False)

        material = self._sim.create_physical_material(0, 0, 0)
        self._scene.add_ground(0, material=material)

        world_width = self.world_size
        self.world_size = world_width
        ranges = np.array([[-world_width, world_width], [-world_width, world_width]])
        self.ranges = ranges
        self.material = material

        DEFAULT_GOAL = [
            [-1.4, 1.4],
            [0., 1.4],
            [1.4, 1.4],
        ]

        for i in range(self.n_block):
            self.blocks.append(
                self.add_articulation(
                    i*self.block_size*3, 0.0, self.block_size, COLORS[i],
                    ranges, f"block{i}", friction=0,
                    damping=5000, material=material)
            )
            self.goal_vis.append(
                self.add_box(*DEFAULT_GOAL[i],
                             (self.block_size, self.block_size, 0.1),
                             np.array(COLORS[i]) * 1.2, f'goal{i}', True, False))
        self.objects = self.blocks + self.goal_vis