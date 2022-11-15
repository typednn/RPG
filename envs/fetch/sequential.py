import gym
import numpy as np
import torch

from .construction import FetchBlockConstructionEnv
from tools.config import Configurable
import gym

class SequentialStack(Configurable, gym.Env):
    # fixed high-level policy, train-low-level directly
    def __init__(self, cfg=None,
                 num_blocks=2,
                 action_range=0.3,
                 frameskip=1,
                 sparse_scale=1.,
                 contact_scale=1.,
                 dist_scale=10.,
                 n_contact_scale=0.0,
                 random_init=False,
                 random_goal=False,
                 double_stage=False,
                 double_stage_steps=15,
                 incremental_reward=True,
                 substeps=30,
                 norm=2,
                 include_step_in_obs=True,
                 distance_threshold=0.05,
                 stop_dist_reward=False):
        super(SequentialStack, self).__init__()
        self.substeps = substeps
        self.sparse_scale = sparse_scale
        self.contact_scale = contact_scale
        self.stop_dist_reward = stop_dist_reward
        self._max_episode_steps = num_blocks * self.substeps
        self.frameskip = frameskip
        self.action_range = action_range
        self.distance_threshold = distance_threshold

        self.stage_id = 0
        self.steps = 0
        self.num_blocks = num_blocks
        self.dist_scale = dist_scale
        self.n_contact_scale = n_contact_scale
        self.random_init = random_init
        self.incremental_reward = incremental_reward
        self.norm = norm
        self.random_goal = random_goal
        self.include_step_in_obs = include_step_in_obs

        reward_type = 'incremental'
        obs_type = 'dictstate'
        render_size = 42
        stack_only = True
        case = 'Singletower'

        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }

        for i in range(num_blocks):
            initial_qpos[F"object{i}:joint"] = [1.25, 0.53, .4 + i * .06, 1., 0., 0., 0.]

        kwargs = {
            'reward_type': reward_type,
            'initial_qpos': initial_qpos,
            'num_blocks': num_blocks,
            'obs_type': obs_type,
            'render_size': render_size,
            'stack_only': stack_only,
            'case': case
        }

        self.env = FetchBlockConstructionEnv(**kwargs)
        self.action_space = gym.spaces.Box(-1, 1, (4,))
        len_obs = self.env.observation_space['observation'].shape[0]

        self.stage_dim = 1 if not double_stage else 2
        self.double_stage = double_stage
        self.double_stage_steps = double_stage_steps

        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (len_obs + self.num_blocks + (self.num_blocks + 1) * 3
                                                                  + int(self.include_step_in_obs),))
        self.observation_space.shared_dim = 10 + int(self.include_step_in_obs)
        self.observation_space.num_blocks = self.num_blocks
        self.observation_space.goal_dim = 3
        self.observation_space.object_dim = 15 + self.stage_dim
        self.goal = None


    def _get_obs(self, include_stage=True, include_stage_id=False):
        obs = self.obs['observation']

        len_obs = 10
        assert len(obs) == len_obs + self.num_blocks * 15
        robot_state = obs[:len_obs]
        block_states = obs[len_obs:].reshape(self.num_blocks, -1)
        if self.include_step_in_obs:
            robot_state = np.append(robot_state, self.steps/self.substeps)
            len_obs += 1
        # if include_stage:
        #     stages = np.zeros((self.num_blocks, self.stage_dim))
        #     stages[self.obj_id, self.stage_id] = 1
        #     block_states = np.concatenate((block_states, stages), 1)
        # if include_stage_id:
        subgoal_dists = self.subgoal_distances(self.obs['achieved_goal'], self.env.goal)
        dist = np.array(subgoal_dists) <= self.distance_threshold
        block_states = np.concatenate((block_states, dist[:, None]), 1)
        obs = np.concatenate((robot_state, block_states.reshape(-1), self.obs['desired_goal']))
        return obs

    def reset(self):
        # we fix consider a determinist
        if not self.random_init or self.goal is None:
            self.env.seed(1)

        sim_config = dict(
            init_pos = dict(
                object0 = [1.14926771, 0.90516281],
                object1 = [1.42368875, 0.67464874],
            )
        )
        # sim_config = {}
        self.env.reset(self.goal is None or self.random_goal, **sim_config)
        self.goal = True # sampled goal

        self.obs = self.env.unwrapped._get_obs()
        self.stage_id = 0
        self.steps = 0
        return self._get_obs()

    # def contact_dist(self):
    #     target_pose = np.copy(self.obs['achieved_goal'][self.obj_id*3:(self.obj_id+1)*3][:])
    #     target_pose[2] += 0.02

    #     return np.linalg.norm(self.obs['achieved_goal'][-3:] - target_pose, self.norm)

    def subgoal_distances(self, achieved, goal):
        out = achieved[:-3] - self.env.goal[:-3]
        out = out.reshape(-1, 3)
        return np.linalg.norm(out, self.norm, axis=1)

    # def get_n_contact(self):
    #     sim = self.env.sim
    #     left=0
    #     right=0
    #     obj_name = f'object{self.obj_id}'
    #     for i in range(sim.data.ncon):
    #         contact = sim.data.contact[i]
    #         a, b = sim.model.geom_id2name(contact.geom1), sim.model.geom_id2name(contact.geom2)
    #         #print('contact', i)
    #         #print('dist', contact.dist)
    #         if a is None or b is None:
    #             continue
    #         if 'finger' in a:
    #             a, b = b, a
    #         if a == obj_name:
    #             if 'l_gripper' in b :
    #                 left = 1
    #             else:
    #                 right = 1
    #     return left + right

    def f(self, x):
        return -(x + np.log(x+0.005))

    # def compute_prev_reward(self):
    #     obj_goal = np.copy(self.env.goal[self.obj_id*3:self.obj_id*3+3])
    #     if self.incremental_reward:
    #         self.prev_dist = np.linalg.norm(self.obs['achieved_goal'][self.obj_id * 3:self.obj_id * 3 + 3] - obj_goal, self.norm)
    #         self.prev_contact = self.contact_dist()

    def compute_reward_info(self, info):
        # obj_goal = np.copy(self.env.goal[self.obj_id*3:self.obj_id*3+3])
        subgoal_dists = self.subgoal_distances(self.obs['achieved_goal'], self.env.goal)
        # contact_dist = self.contact_dist()
        contact_dists = []
        not_reached = np.array(subgoal_dists) > self.distance_threshold

        for obj_id in range(self.num_blocks):
            target_pose = np.copy(self.obs['achieved_goal'][obj_id*3:(obj_id+1)*3][:])
            target_pose[2] += 0.02

            to_contact = np.linalg.norm(self.obs['achieved_goal'][-3:] - target_pose, self.norm)
            if not_reached[obj_id]:
                pass
            else:
                to_contact = 0.

            contact_dists.append(to_contact) # hand ..

            # obj_goal = np.copy(self.env.goal[obj_id*3:obj_id*3+3])
            # goal_dists.append(np.linalg.norm(self.obs['achieved_goal'][obj_id * 3:obj_id * 3 + 3] - obj_goal, self.norm))

        # print(contact_dists)


        r = - not_reached.sum()
        if len(contact_dists) > 0:
            contact_reward = - np.min(contact_dists)
        else:
            contact_dists = 0.
        dist_reward = - subgoal_dists.sum()

        info['success'] = (1-not_reached).sum()

        # contact_reward = -contact_dist
        # dist_reward = -goal_dist

        # if self.incremental_reward:
        #     dist_reward = (dist_reward + self.prev_dist) * 10
        #     contact_reward = (contact_reward + self.prev_contact) * 10

        # if self.double_stage:
        #     if self.stage_id == 0:
        #         dist_reward = 0.0
        #     else:
        #         contact_reward *= 0.3
        # else:
        #     contact_reward *= 0.3

        # if self.stop_dist_reward and subgoal_dists[self.obj_id]<=self.distance_threshold:
        #     dist_reward = 0

        reward = ((r + self.num_blocks) * self.sparse_scale # bias ..
                  + contact_reward * self.contact_scale
                  + dist_reward * self.dist_scale)
        return reward

    def step(self, action):
        action = np.array(action).clip(-1, 1)
        # print('prev achieved', self.obs['achieved_goal'][self.obj_id * 3:(self.obj_id+1)*3])
        # self.compute_prev_reward()
        self.obs, r, _, info = self.env.step(action)
        reward = self.compute_reward_info(info)

        obs = self._get_obs()

        # # don't write below
        # stage_step = (self.steps + 1) % self.substeps
        # if stage_step == 0:
        #     self.obj_id = min(self.obj_id+1, self.num_blocks-1)
        #     self.stage_id = 0
        # if stage_step == self.double_stage_steps and self.double_stage:
        #     self.stage_id = 1

        self.steps += 1

        # done = (self.steps>=self._max_episode_steps)
        return obs, reward, False, info

    def render(self, mode='human'):
        return self.env.render(mode)

    # def state_vector(self):
    #     return np.concatenate([self.env.get_state(), [self.obj_id, self.steps]])

    # def set_state(self, state):
    #     self.obj_id = int(state[-2])
    #     self.steps = int(state[-1])
    #     self.env.set_state(state[:-2])