import numpy as np
from .open_carbinet import OpenCabinetDoorEnv, MobilePandaSingleArm
from mani_skill2.agents.robots.panda import Panda


class FixArm(OpenCabinetDoorEnv): 
    # return information about the environment

    def __init__(self, *args, **kwargs):
        config = {
            'reward_mode': 'dense', 'obs_mode': 'state', 'model_ids': '1018', 'fixed_target_link_idx': 1
        }
        config.update(kwargs)
        super().__init__(*args, **config)

        self.joints = self.agent.robot.get_active_joints()
        assert len(self.agent.robot.get_qpos()) == len(self.joints)
        print([i.name for i in self.joints])

        qpos = self.agent.robot.get_qpos()
        for i in range(4):
            self.joints[i].set_limits([[qpos[i]-1e-5, qpos[i] + 1e-5]])

        self.sample_anchor_points()


    def _initialize_robot(self):
        # Base position
        # The forward direction of cabinets is -x.
        center = np.array([1, 0.0])
        #dist = self._episode_rng.uniform(1.6, 1.8)
        #theta = self._episode_rng.uniform(0.9 * np.pi, 1.1 * np.pi)
        dist = 1.6
        # theta = 1.
        theta = np.pi
        direction = np.array([np.cos(theta), np.sin(theta)])
        xy = center + direction * dist

        # Base orientation
        noise_ori = self._episode_rng.uniform(-0.05 * np.pi, 0.05 * np.pi)
        #ori = (theta - np.pi) + noise_ori
        ori = 0.

        h = 1e-4
        arm_qpos = np.array([0, 0, 0, -1.5, 0, 3, 0.78, 0.02, 0.02])

        qpos = np.hstack([xy, ori, h, arm_qpos])
        self.agent.reset(qpos)
        

    def wrap_obs(self):
        raise NotImplementedError

    def decode_obs(self, obs):
        raise NotImplementedError

    def sample_anchor_points(self, N=10000):
        state = np.random.get_state()
        np.random.seed(0)
        qlimits = np.array([i.get_limits() for i in self.joints])[:, 0]
        anchors = []
        for i in range(len(qlimits)):
            random_qpos = np.random.uniform(qlimits[i, 0], qlimits[i, 1], (N,))
            anchors.append(random_qpos)
        self.anchors = np.stack(anchors, axis=1)
        np.random.set_state(state)

    def show_random_anchor(self):
        x = self.anchors[np.random.randint(len(self.anchors))]
        self.agent.robot.set_qpos(x)
        return self.render(mode='rgb_array')

    def _load_agent(self):
        self.agent = MobilePandaSingleArm(
            self._scene, self._control_freq, self._control_mode, fix_root_link=True
        )