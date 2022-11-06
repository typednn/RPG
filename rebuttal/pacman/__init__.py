from .pacman import PacManEnv
from .antman import AntManEnv
from .pacman_single_layer import PacManEnvSingleLayer
from ..env_builder import EnvBuilder
from ..hiro_env import HIROEnv
from gym.wrappers import TimeLimit

class PacMan(EnvBuilder):
    def __init__(self,
                 cfg=None,
                 size=4,
                 random_maze=True,
                 random_goal=True,
                 action_scale=0.25,
                 substeps=5,
                 low_reward_type='dense', # use dense by default.
                 max_timesteps=100,
                 include_low_obs=1.,
                 penalty=0.
                 ):
        EnvBuilder.__init__(self, cfg)
        env = PacManEnv(size, size,
                        random_maze,
                        random_goal,
                        action_scale=action_scale,
                        reward_type=low_reward_type,
                        include_low_obs=include_low_obs,
                        penalty=penalty)
        self.env = HIROEnv(env, substeps, max_timesteps, high_reward_type='last')


class AntMan(EnvBuilder):
    def __init__(self,
                 cfg=None,
                 size=4,
                 random_maze=True,
                 random_goal=True,
                 substeps=20,
                 low_reward_type='dense',
                 max_timesteps=400,
                 include_low_obs=0.2,
                 penalty=0.,
                 ):
        EnvBuilder.__init__(self, cfg)
        env = AntManEnv(size, size,
                        random_maze,
                        random_goal,
                        reward_type=low_reward_type,
                        include_low_obs=include_low_obs,
                        penalty=penalty)
        self.env = HIROEnv(env, substeps, max_timesteps, high_reward_type='last')


class OneMan(EnvBuilder):
    # one layer pacman..
    def __init__(self, cfg=None, size=4,
                 random_maze=True,
                 random_goal=True,
                 max_timesteps=20,
                 penalty=0.):
        EnvBuilder.__init__(self, cfg)
        env = PacManEnvSingleLayer(size, size, random_maze, random_goal, max_timesteps, penalty=penalty)
        self.env = TimeLimit(env, max_timesteps)