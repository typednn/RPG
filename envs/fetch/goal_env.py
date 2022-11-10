import os
import cv2
import numpy as np
import torch
import copy
import pickle
from typing import Optional, TypeVar, Type, Union

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from .sequential import *

class GoalRelabeler:
    def __init__(self,
                 reward_type,
                 num_blocks,
                 incremental_reward,
                 sparse_scale,
                 contact_scale,
                 dist_scale,
                 norm,
                 threshold,
                 contact_stage2_scale=0.3,
                 goal_stage1_scale=0.3,
                 ):
        self.reward_type = reward_type
        self.num_blocks = num_blocks

        self.incremental_reward = incremental_reward
        self.sparse_scale = sparse_scale
        self.contact_scale = contact_scale
        self.dist_scale = dist_scale
        self.norm = norm
        self.threshold = threshold

        self.contact_stage2_scale = contact_stage2_scale
        self.goal_stage1_scale = goal_stage1_scale

    def get_stage_batch(self, state):
        stage = (state[:, 13:-(self.num_blocks+1)*3].reshape(len(state), self.num_blocks, -1))[:, :, -2:]
        batch_id, obj_id, stage_id = np.where(stage)
        return batch_id, obj_id, stage_id

    def get_stage(self, state):
        stage = (state[13:-(self.num_blocks+1)*3].reshape(self.num_blocks, -1))[:, -2:]
        x, y = np.where(stage)
        obj_id, stage_id = x[0], y[0]
        return obj_id, stage_id

    def relabel_batch(self, cur_state, next_state, future_state=None):
        batch_id, obj_id, stage_id = self.get_stage_batch(cur_state)
        start_id = -(self.num_blocks + 1) * 3

        if future_state is not None:
            cur_state = np.copy(cur_state)
            next_state = np.copy(next_state)

            mask = np.float32(stage_id == 0)[:, None]

            #cur_state[:, 10:13] = cur_state[:, 10:13] * (1-mask) + (future_state[:, :3] - cur_state[:, :3]) * mask
            #next_state[:, 10:13] = next_state[:, 10:13] * (1-mask) + (future_state[:, :3] - next_state[:, :3]) * mask
            cur_state[:, 10:13] = future_state[:, :3] - cur_state[:, :3]
            next_state[:, 10:13] = future_state[:, :3] - next_state[:, :3]
            goal = future_state[:, 13:start_id].reshape(len(future_state), self.num_blocks, -1)[batch_id, obj_id, :3]

            mask = mask[:, 0]
            for j in range(3):
                cur_state[batch_id, start_id+obj_id*3+j] = cur_state[batch_id, start_id+obj_id*3+j] * mask + goal[:, j] * (1-mask)
                next_state[batch_id, start_id+obj_id*3+j] = next_state[batch_id, start_id+obj_id*3+j] * mask + goal[:, j] * (1-mask)
                #cur_state[batch_id, start_id+obj_id*3+j] = goal[:, j]
                #next_state[batch_id, start_id+obj_id*3+j] = goal[:, j]

        goal = next_state[:, start_id:-3].reshape(len(goal), self.num_blocks, -1)
        contact_dist = np.linalg.norm(next_state[:, 10:13], self.norm, axis=1)
        block2goal = np.linalg.norm((next_state[:, 13:start_id].reshape(len(goal), self.num_blocks, -1)[:, :, :3]
                                     - goal).reshape(len(goal), -1, 3), self.norm, axis=2)
        goal_dist = block2goal[batch_id, obj_id]
        # print(obj_id, stage_id, block2goal, contact_dist, goal_dist)

        if self.incremental_reward:
            contact_dist -= np.linalg.norm(cur_state[:, 10:13], self.norm, axis=1)
            goal_dist -= \
                np.linalg.norm((cur_state[:, 13:start_id].reshape(len(goal), self.num_blocks, -1)[:, :, :3]
                                - goal).reshape(len(goal), -1, 3), self.norm, axis=2)[batch_id, obj_id]
            contact_dist *= 10
            goal_dist *= 10

        mask = np.float32(stage_id == 0)
        contact_dist = contact_dist * mask + contact_dist * (1-mask) * self.contact_stage2_scale #if 0, 1. else 0.3
        goal_dist = goal_dist * mask * self.goal_stage1_scale + goal_dist * (1-mask) # only activated at stage 1..

        reward = - contact_dist * self.contact_scale \
                 - goal_dist * self.dist_scale \
                 + (block2goal < self.threshold).sum(axis=1) * self.sparse_scale
        return cur_state, next_state, reward

    def __call__(self, cur_state, next_state, future_state=None):
        # similar to relabel two
        obj_id, stage_id = self.get_stage(cur_state)
        start_id = -(self.num_blocks + 1) * 3
        if future_state is not None:
            cur_state = np.copy(cur_state)
            next_state = np.copy(next_state)


            # no matter what, move to the goal direction ..
            cur_state[10:13] = future_state[:3] - cur_state[:3]  # this is the goal stage
            next_state[10:13] = future_state[:3] - next_state[:3]  # this is the goal stage

            if stage_id == 1:
                goal = future_state[13:start_id].reshape(self.num_blocks, -1)[obj_id, :3]
                cur_state[start_id+obj_id*3:start_id+obj_id*3+3] = goal
                next_state[start_id+obj_id*3:start_id+obj_id*3+3] = goal

        goal = next_state[start_id:-3].reshape(self.num_blocks, -1)
        contact_dist = np.linalg.norm(next_state[10:13], self.norm)
        block2goal = np.linalg.norm((next_state[13:start_id].reshape(self.num_blocks, -1)[:, :3]
                                     - goal).reshape(-1, 3), self.norm, axis=1)
        goal_dist = block2goal[obj_id]
        # print(obj_id, stage_id, block2goal, contact_dist, goal_dist)
        #print('+', goal_dist, next_state[start_id:-3].reshape(self.num_blocks, -1), goal)

        self.goal_dist = goal_dist

        if self.incremental_reward:
            contact_dist -= np.linalg.norm(cur_state[10:13], self.norm)
            goal_dist -= \
                np.linalg.norm((cur_state[13:start_id].reshape(self.num_blocks, -1)[:, :3]
                                - goal).reshape(-1, 3), self.norm, axis=1)[obj_id]
            contact_dist *= 10
            goal_dist *= 10

        if stage_id == 0:
            goal_dist *= self.goal_stage1_scale
        else:
            contact_dist *= self.contact_stage2_scale
        #if goal_dist != 0:
        #print(stage_id, contact_dist, goal_dist)
        #print('-', goal_dist)

        reward = - contact_dist * self.contact_scale \
                 - goal_dist * self.dist_scale \
                 + (block2goal < self.threshold).sum() * self.sparse_scale
        return cur_state, next_state, reward

class GoalEnv(SequentialStack):
    def __init__(self, cfg=None,
                 reward_type='dense',
                 random_goal=True,
                 random_init=True,
                 incremental_reward=True,
                 sparse_scale=5.,
                 norm=1):
        super(GoalEnv, self).__init__(cfg=cfg, double_stage=True)
        self.reward_type = reward_type
        #assert self.random_goal and self.random_init

        # add goal
        new_observation_space = gym.spaces.Box(-np.inf, np.inf, (self.observation_space.shape[0]+3,))
        new_observation_space.shared_dim = 10 + 3 # add goal for the gripper ..
        new_observation_space.num_blocks = self.num_blocks
        new_observation_space.goal_dim = 3
        new_observation_space.object_dim = 15 + self.stage_dim
        self.goal_relabel = new_observation_space.goal_relabel \
                          = GoalRelabeler(reward_type, self.num_blocks,
                                                           self.incremental_reward,
                                                           self.sparse_scale,
                                                           self.contact_scale,
                                                           self.dist_scale,
                                                           self.norm,
                                                           self.env.distance_threshold)
        self.observation_space = new_observation_space

    def update_desired_goal(self):
        self.desired_goal = np.copy(self.obs['desired_goal'])
        self.desired_goal[:-3] = self.obs['achieved_goal'][:-3]

    def reset(self):
        #self.desired_goal = np.copy(self.env._sample_goal())
        super(GoalEnv, self).reset()
        #self.update_desired_goal()
        return self._get_obs()

    def _get_obs(self):
        obs = self.obs['observation']

        assert len(obs) == 10 + self.num_blocks * 15
        robot_state = obs[:10]

        achieved_goal = self.obs['achieved_goal']
        target_pose = np.copy(achieved_goal[self.obj_id*3:(self.obj_id+1)*3][:])
        target_pose[2] += 0.02
        robot_state = np.concatenate((robot_state, target_pose - achieved_goal[-3:]))

        block_states = obs[10:].reshape(self.num_blocks, -1)
        stages = np.zeros((self.num_blocks, self.stage_dim))
        stages[self.obj_id, self.stage_id] = 1
        block_states = np.concatenate((block_states, stages), 1)
        #raise NotImplementedError("desired goal seems to not correct ...")
        #desired_goal = np.copy(self.desired_goal)
        #if self.stage_id == 1:
        #    desired_goal[self.obj_id*3:self.obj_id*3+3] = self.env.goal[self.obj_id*3:self.obj_id*3+3]
        desired_goal = np.copy(self.env.goal)
        obs = np.concatenate((robot_state, block_states.reshape(-1), desired_goal))
        return obs

    def step(self, action):
        prev_obs = self._get_obs()
        action = np.array(action).clip(-1, 1)

        self.obs, r, _, info = self.env.step(action)
        dist = self.subgoal_distances(self.obs['achieved_goal'], self.env.goal)[self.obj_id]

        contact = self.contact_dist()

        obs = self._get_obs()
        _, _, reward = self.goal_relabel(prev_obs, obs, None)

        #if self.stage_id == 1:
        #    assert np.allclose(dist, self.goal_relabel.goal_dist), f"{dist} {self.goal_relabel.goal_dist}"

        info['done_bool'] = False
        subgoal_dists = self.subgoal_distances(self.obs['achieved_goal'], self.env.goal)
        info['success'] = (np.array(subgoal_dists) < self.env.distance_threshold).sum()


        # don't write below
        stage_step = self.steps % self.substeps
        if stage_step == 0 and self.steps!=0:
            self.obj_id = min(self.obj_id + 1, self.num_blocks - 1)
            self.stage_id = 0
            # self.update_desired_goal()

        if stage_step == self.double_stage_steps:
            self.stage_id = 1

        self.steps += 1

        done = (self.steps >= self._max_episode_steps)
        return obs, reward, done, info

register(GoalEnv)