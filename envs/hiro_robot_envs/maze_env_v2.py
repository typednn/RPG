# build a maze with less thiner walls... the current one is not very efficient..

import os
import numpy as np

"""Adapted from hiro-robot-envs maze_env.py."""

import os
import tempfile
import xml.etree.ElementTree as ET
import numpy as np
import gym

from . import maze_env_utils

# Directory that contains mujoco xml files.
MODEL_DIR = 'assets'


class MazeEnvV2(gym.Env):
    MODEL_CLASS = None

    MAZE_HEIGHT = None
    MAZE_SIZE_SCALING = None

    def __init__(
            self,
            width=4,
            height=4,
            maze_id=None,
            maze_height=0.5,
            wall_size = 0.05,
            maze_size_scaling=8,
            *args, **kwargs
    ):
        self._maze_id = maze_id
        self.wall_size = wall_size
        self.t = 0

        self.MAZE_HEIGHT = maze_height
        self.MAZE_SIZE_SCALING = maze_size_scaling

        self.width = width
        self.height = height


        model_cls = self.__class__.MODEL_CLASS
        if model_cls is None:
            raise Exception("MODEL_CLASS unspecified!")
        xml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_DIR, model_cls.FILE)
        tree = ET.parse(xml_path)

        worldbody = tree.find(".//worldbody")
        for i in range(width+1):
            for j in range(height+1):
                if i != width:
                    self.add_wall(worldbody, f"wall_{i}_{j}_h", i, j, 0)
                if j != height:
                    self.add_wall(worldbody, f"wall_{i}_{j}_v", i, j, 1)

        torso = tree.find(".//body[@name='torso']")
        geoms = torso.findall(".//geom")
        for geom in geoms:
            if 'name' not in geom.attrib:
                raise Exception("Every geom of the torso must have a name "
                                "defined")

        _, file_path = tempfile.mkstemp(text=True, suffix=".xml")
        tree.write(file_path)

        self.wrapped_env = model_cls(*args, file_path=file_path, **kwargs)

        self._init_poses = self.wrapped_env.model.geom_pos[:].copy()

    def add_wall(self, worldbody, name, x, y, type=0):
        scale = self.MAZE_SIZE_SCALING

        if type == 0:
            pos = "%f %f %f" % ((x + 0.5) * scale, y * scale, self.MAZE_HEIGHT / 2 * scale)
            size = "%f %f %f" % ((0.5 + self.wall_size) * scale, self.wall_size * scale, self.MAZE_HEIGHT / 2 * scale)
        else:
            pos = "%f %f %f" % (x * scale, (y + 0.5) * scale, self.MAZE_HEIGHT / 2 * scale)
            size = "%f %f %f" % (self.wall_size * scale, (0.5 + self.wall_size) * scale, self.MAZE_HEIGHT / 2 * scale)

        ET.SubElement(
            worldbody, "geom", name=name, pos=pos, size=size,
            type="box", material="", contype="1", conaffinity="1", rgba="0.4 0.4 0.4 1",
        )

    def set_map(self, map):
        poses = self._init_poses.copy()

        for i in range(self.width):
            for j in range(self.height):
                if not map[0, j, i]:
                    name = f"wall_{i}_{j}_h"
                    id = self.wrapped_env.model.geom_name2id(name)
                    poses[id][:2] -= 100
                if not map[2, j, i]:
                    name = f"wall_{i}_{j}_v"
                    id = self.wrapped_env.model.geom_name2id(name)
                    poses[id][:2] -= 100
        self.wrapped_env.model.geom_pos[:] = poses
        self.wrapped_env.sim.forward()

    def _get_obs(self):
        return np.concatenate([self.wrapped_env._get_obs(),
                               [self.t * 0.001]])

    def reset(self):
        self.t = 0
        self.wrapped_env.reset()
        return self._get_obs()

    @property
    def viewer(self):
        return self.wrapped_env.viewer

    def render(self, *args, **kwargs):
        return self.wrapped_env.render(*args, **kwargs)

    @property
    def observation_space(self):
        shape = self._get_obs().shape
        high = np.inf * np.ones(shape)
        low = -high
        return gym.spaces.Box(low, high)

    @property
    def action_space(self):
        return self.wrapped_env.action_space

    def step(self, action):
        self.t += 1
        inner_next_obs, inner_reward, done, info = self.wrapped_env.step(action)
        next_obs = self._get_obs()
        done = False
        return next_obs, inner_reward, done, info
