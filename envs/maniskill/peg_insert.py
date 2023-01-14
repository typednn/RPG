

from collections import OrderedDict

import numpy as np
import sapien.core as sapien
from sapien.core import Pose
from transforms3d.euler import euler2quat

from mani_skill2.utils.registration import register_gym_env
from mani_skill2.utils.sapien_utils import hex2rgba, look_at, vectorize_pose

from mani_skill2.envs.assembly.base_env import StationaryManipulationEnv


class PegInsert(StationaryManipulationEnv):
    _clearance = 0.003

    def __init__(self, *args, obs_dim=8, reward_type='sparse', **kwargs):
        self.obs_dim = obs_dim
        self.reward_type = reward_type
        super().__init__(*args, **kwargs)

    def reset(self, reconfigure=True, **kwargs):
        return super().reset(reconfigure=reconfigure, **kwargs)

    def _build_box_with_hole(
        self, inner_radius, outer_radius, depth, center=(0, 0), name="box_with_hole"
    ):
        builder = self._scene.create_actor_builder()
        thickness = (outer_radius - inner_radius) * 0.5
        # x-axis is hole direction
        half_center = [x * 0.5 for x in center]
        half_sizes = [
            [depth, thickness - half_center[0], outer_radius],
            [depth, thickness + half_center[0], outer_radius],
            [depth, outer_radius, thickness - half_center[1]],
            [depth, outer_radius, thickness + half_center[1]],
        ]
        offset = thickness + inner_radius
        poses = [
            Pose([0, offset + half_center[0], 0]),
            Pose([0, -offset + half_center[0], 0]),
            Pose([0, 0, offset + half_center[1]]),
            Pose([0, 0, -offset + half_center[1]]),
        ]

        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#FFD289"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5

        for (half_size, pose) in zip(half_sizes, poses):
            builder.add_box_collision(pose, half_size)
            builder.add_box_visual(pose, half_size, material=mat)
        return builder.build_static(name)

    def _load_actors(self):
        self._add_ground()

        # peg
        # length, radius = 0.1, 0.02
        length = self._episode_rng.uniform(0.075, 0.125)
        radius = self._episode_rng.uniform(0.015, 0.025)
        builder = self._scene.create_actor_builder()
        builder.add_box_collision(half_size=[length, radius, radius])

        # peg head
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EC7357"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        # peg tail
        mat = self._renderer.create_material()
        mat.set_base_color(hex2rgba("#EDF6F9"))
        mat.metallic = 0.0
        mat.roughness = 0.5
        mat.specular = 0.5
        builder.add_box_visual(
            Pose([-length / 2, 0, 0]),
            half_size=[length / 2, radius, radius],
            material=mat,
        )

        self.peg = builder.build("peg")
        self.peg_head_offset = Pose([length, 0, 0])
        self.peg_half_size = np.float32([length, radius, radius])

        # box with hole
        center = 0.5 * (length - radius) * self._episode_rng.uniform(-1, 1, size=2)
        inner_radius, outer_radius, depth = radius + self._clearance, length, length
        self.box = self._build_box_with_hole(
            inner_radius, outer_radius, depth, center=center
        )
        self.box_hole_offset = Pose(np.hstack([0, center]))
        self.box_hole_radius = inner_radius

    def _initialize_actors(self):
        #xy = self._episode_rng.uniform([-0.1, -0.3], [0.1, 0])
        xy = np.array([0, -0.15])
        pos = np.hstack([xy, self.peg_half_size[2]])
        ori = np.pi / 2 #+ self._episode_rng.uniform(-np.pi / 3, np.pi / 3)
        quat = euler2quat(0, 0, ori)
        self.peg.set_pose(Pose(pos, quat))

        #xy = self._episode_rng.uniform([-0.05, 0.2], [0.05, 0.4])
        xy = np.array([0., 0.3])
        pos = np.hstack([xy, self.peg_half_size[0]])
        ori = np.pi / 2 # + self._episode_rng.uniform(-np.pi / 8, np.pi / 8)
        quat = euler2quat(0, 0, ori)
        self.box.set_pose(Pose(pos, quat))

    def _initialize_agent(self):
        if self.robot_uuid == "panda":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            qpos[:-2] += self._episode_rng.normal(
                0, self.robot_init_qpos_noise, len(qpos) - 2
            )
            self.agent.reset(qpos)
            self.agent.robot.set_pose(Pose([-0.615, 0, 0]))
        else:
            raise NotImplementedError(self.robot_uuid)

    @property
    def peg_head_pos(self):
        return self.peg.pose.transform(self.peg_head_offset).p

    @property
    def peg_head_pose(self):
        return self.peg.pose.transform(self.peg_head_offset)

    @property
    def box_hole_pose(self):
        return self.box.pose.transform(self.box_hole_offset)

    def _initialize_task(self):
        self.goal_pos = self.box_hole_pose.p  # goal of peg head inside the hole
        # NOTE(jigu): The goal pose is computed based on specific geometries used in this task.
        # Only consider one side
        self.goal_pose = (
            self.box.pose * self.box_hole_offset * self.peg_head_offset.inv()
        )
        # self.peg.set_pose(self.goal_pose)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                peg_pose=vectorize_pose(self.peg.pose),
                peg_half_size=self.peg_half_size,
                box_hole_pose=vectorize_pose(self.box_hole_pose),
                box_hole_radius=self.box_hole_radius,
            )
        return obs

    def has_peg_inserted(self):
        # Only head position is used in fact
        peg_head_pose = self.peg.pose.transform(self.peg_head_offset)
        box_hole_pose = self.box_hole_pose
        peg_head_pos_at_hole = (box_hole_pose.inv() * peg_head_pose).p
        # x-axis is hole direction
        x_flag = -0.015 <= peg_head_pos_at_hole[0]
        y_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[1] <= self.box_hole_radius
        )
        z_flag = (
            -self.box_hole_radius <= peg_head_pos_at_hole[2] <= self.box_hole_radius
        )
        return (x_flag and y_flag and z_flag), peg_head_pos_at_hole

    def evaluate(self, **kwargs) -> dict:
        success, peg_head_pos_at_hole = self.has_peg_inserted()
        return dict(success=success, peg_head_pos_at_hole=peg_head_pos_at_hole)

    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0

        if info["success"]:
            return 25.0

        # grasp pose rotation reward
        tcp_pose_wrt_peg = self.peg.pose.inv() * self.tcp.pose
        tcp_rot_wrt_peg = tcp_pose_wrt_peg.to_transformation_matrix()[:3, :3]
        gt_rot_1 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        gt_rot_2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        grasp_rot_loss_fxn = lambda A: np.arcsin(
            np.clip(1 / (2 * np.sqrt(2)) * np.sqrt(np.trace(A.T @ A)), 0, 1)
        )
        grasp_rot_loss = np.minimum(
            grasp_rot_loss_fxn(gt_rot_1 - tcp_rot_wrt_peg),
            grasp_rot_loss_fxn(gt_rot_2 - tcp_rot_wrt_peg),
        ) / (np.pi / 2)
        rotated_properly = grasp_rot_loss < 0.2
        reward += 1 - grasp_rot_loss

        gripper_pos = self.tcp.pose.p
        tgt_gripper_pose = self.peg.pose
        offset = sapien.Pose(
            [-0.06, 0, 0]
        )  # account for panda gripper width with a bit more leeway
        tgt_gripper_pose = tgt_gripper_pose.transform(offset)
        if rotated_properly:
            # reaching reward
            gripper_to_peg_dist = np.linalg.norm(gripper_pos - tgt_gripper_pose.p)
            reaching_reward = 1 - np.tanh(
                4.0 * np.maximum(gripper_to_peg_dist - 0.015, 0.0)
            )
            # reaching_reward = 1 - np.tanh(10.0 * gripper_to_peg_dist)
            reward += reaching_reward

            # grasp reward
            is_grasped = self.agent.check_grasp(
                self.peg, max_angle=20
            )  # max_angle ensures that the gripper grasps the peg appropriately, not in a strange pose
            if is_grasped:
                reward += 2.0

            # pre-insertion award, encouraging both the peg center and the peg head to match the yz coordinates of goal_pose
            pre_inserted = False
            if is_grasped:
                peg_head_wrt_goal = self.goal_pose.inv() * self.peg_head_pose
                peg_head_wrt_goal_yz_dist = np.linalg.norm(peg_head_wrt_goal.p[1:])
                peg_wrt_goal = self.goal_pose.inv() * self.peg.pose
                peg_wrt_goal_yz_dist = np.linalg.norm(peg_wrt_goal.p[1:])
                if peg_head_wrt_goal_yz_dist < 0.01 and peg_wrt_goal_yz_dist < 0.01:
                    pre_inserted = True
                    reward += 3.0
                pre_insertion_reward = 3 * (
                    1
                    - np.tanh(
                        0.5 * (peg_head_wrt_goal_yz_dist + peg_wrt_goal_yz_dist)
                        + 4.5
                        * np.maximum(peg_head_wrt_goal_yz_dist, peg_wrt_goal_yz_dist)
                    )
                )
                reward += pre_insertion_reward

            # insertion reward
            if is_grasped and pre_inserted:
                peg_head_wrt_goal_inside_hole = (
                    self.box_hole_pose.inv() * self.peg_head_pose
                )
                insertion_reward = 5 * (
                    1 - np.tanh(5.0 * np.linalg.norm(peg_head_wrt_goal_inside_hole.p))
                )
                reward += insertion_reward
        else:
            reward = reward - 10 * np.maximum(
                self.peg.pose.p[2] + self.peg_half_size[2] + 0.01 - self.tcp.pose.p[2],
                0.0,
            )
            reward = reward - 10 * np.linalg.norm(
                tgt_gripper_pose.p[:2] - self.tcp.pose.p[:2]
            )

        return reward

    def _setup_cameras(self):
        super()._setup_cameras()
        self.render_camera.set_local_pose(look_at([1.0, -1.0, 0.8], [0.0, 0.0, 0.5]))
        self._cameras["base_camera"].set_local_pose(
            look_at([0, -0.3, 0.2], [0, 0, 0.1])
        )

    def set_state(self, state):
        super().set_state(state)
        # NOTE(xuanlin): This way is specific to how we compute goals.
        # The general way is to handle variables explicitly
        self._initialize_task()

    def _add_ground(self, altitude=0.0, render=True):
        # half_size = 0.75
        half_size = 10000.0
        render=True
        if render:
            rend_mtl = self._renderer.create_material()
            rend_mtl.base_color = [0.4, 0.5, 0.7, 1]
            rend_mtl.metallic = 0.0
            rend_mtl.roughness = 0.9
            rend_mtl.specular = 0.8
        else:
            rend_mtl = None
        ab = self._scene.create_actor_builder()
        # if render:
        #     ab.add_box_visual(half_size=[half_size, half_size, .0001], material=rend_mtl)
        ab.add_box_collision(half_size=[half_size, half_size, .0001])
        box = ab.build_static()
        # box.set_pose(sapien.Pose(p=[-0.2,0,altitude-.0001])) 
        box.set_pose(sapien.Pose(p=[0.,0.,altitude-.0001])) 
        return box

    def _get_obs(self, obs):
        def devectorize_pose(pose):
            return Pose(pose[:3], pose[3:])
        tcp_pose = devectorize_pose(obs[25:25+7])
        peg_pose = devectorize_pose(obs[25+7:25+7+7])
        box_hole_pos = devectorize_pose(obs[25+17: 25 + 17 + 7])

        # A = tcp_pose * peg_pose.inv()
        # point in tcp frame first to world and the transform to the peg frame ..
        from envs import utils 
        #a =  peg_pose.inv() * tcp_pose
        #b =  peg_pose.inv() * box_hole_pos
        a = peg_pose.p - tcp_pose.p
        b = peg_pose.p - box_hole_pos.p


        if not hasattr(self, 'embedder') and self.obs_dim > 0:
            self.embedder, d = utils.get_embeder_np(self.obs_dim, 10) # only for the differences between objects ..

        if self.obs_dim == 0:
            return obs
        inp = np.concatenate((utils.symlog(a/0.4), utils.symlog(b/0.4), utils.symlog(peg_pose.p))) # tcp opened ..
        inp = self.embedder(inp)
        return np.concatenate((utils.symlog(obs) * 0.2, inp))


    def reset(self, reconfigure=False):
        # simple trick
        obs = super().reset(reconfigure=reconfigure)

        # print('tcp pose', tcp_pose)
        # print('peg pose', peg_pose)
        # print('box_hole_pos', box_hole_pos)
        # print(obs[25:])
        # exit(0)
        return self._get_obs(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        done = False
        if self.reward_type == 'sparse':
            reward = float(info['success'] > 0)
        else:
            reward = reward * 0.1
        return self._get_obs(obs), reward, done, info

    def _render_traj_rgb(self, traj, occ_val=False, history=None, verbose=True, **kwargs):
        # don't count occupancy now ..
        # from .. import utils
        # high = 0.4
        # obs = utils.extract_obs_from_tarj(traj) / 0.05
        # stick = obs[..., 4:7]
        # handle = obs[..., 11:14]
        # outs = dict(occ=utils.count_occupancy(handle - stick, -high, high, 0.02))
        # history = utils.update_occupancy_with_history(outs, history)
        history = {}
        output = {
            'background': {},
            'history': history,
            'image': {},
            'metric': {k: (v > 0.).mean() for k, v in history.items()},
        }
        return output


if __name__ == '__main__':
    from tools.utils import animate
    env = PegInsert()
    print(env.observation_space)
    print(env.action_space)

    images = []
    for i in range(1000):
        if i% 100 == 0:
            env.reset()
        env.step(env.action_space.sample())
        images.append(env.render(mode='rgb_array'))
    
    animate(images)