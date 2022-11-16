import torch
import numpy as np
from mpm.simulator import MPMSimulator
from tools import Configurable, as_builder, merge_inputs
from mpm.utils import rigid_body_motion
from pytorch3d.transforms import quaternion_apply


class Tool(Configurable): 
    n_bodies = 2

    def __init__(
        self, simulator,
        cfg=None,
        friction=20.,
        K=0.0, # stiffness
        color=(0.3, 0.3, 0.3),
        softness=666.,

        lv=(0.01, 0.01, 0.01), # linear velocity
        av=(0.01, 0.01, 0.01), # angular velocity
    ):
        super().__init__()

        # mu is the friction
        self.simulator: MPMSimulator = simulator
        self.name: str = None
        self.substeps = self.simulator.substeps

        self.init()

    def init(self):
        self.tool_lists = {k: [] for k in
                           ['types', 'softness', 'mu', 'K', 'args', 'pos', 'rot', 'action_scales']}

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 7)

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6,)

    def check_action(self, action):
        assert action.max() < 1.1 and action.min() > -1.1
        assert action.shape == self.get_action_shape(), "{} != {}".format(action.shape, self.get_action_shape())

    def add_tool(self, type, args, pos=None, rot=None, **kwargs):
        # print(list(self.tool_lists.keys()))
        def append(key, val):
            self.tool_lists[key].append(val)

        if isinstance(type, str):
            type = {
                'Box': 0, # (x, y, z)
                'Capsule': 1, # (r, h)
            }[type]

        params = np.zeros(4)
        for idx, i in enumerate(args):
            params[idx] = i

        cfg = merge_inputs(self._cfg, **kwargs)
        append('types', type)
        # print(cfg.softness)
        append('softness', cfg.softness)
        append('mu', cfg.friction)
        append('K', cfg.K)
        append('args', params)
        append('pos', (0., 0., 0.) if pos is None else pos)
        append('rot', (1., 0., 0., 0.) if rot is None else rot)
        append('action_scales', cfg.lv + cfg.av)

    def reset(self, cfg=None):
        if cfg is not None:
            self._cfg.merge_from_other_cfg(cfg)
            self.init()

        self.simulator.n_bodies = len(self.tool_lists['types'])
        assert self.simulator.n_bodies == self.n_bodies
        assert self.simulator.n_bodies <= self.simulator.MAX_N_BODIES, f"{self.simulator.n_bodies}, {self.simulator.MAX_N_BODIES}"
        self.simulator.init_bodies(**self.tool_lists)

    def set_qpos(self, cur, val):
        """
        for i in range(cur+1, cur+self.substeps+1):
            if i > len(self.qpos_list):
                self.qpos_list.append(val[i-cur-1])
            else:
        self.qpos_list[cur+1:cur+self.substeps+1] = val
        """
        if cur + 1 == len(self.qpos_list):
            # for backward..
            self.qpos_list = torch.concat((self.qpos_list, val))
        else:
            assert cur == 0
            self.qpos_list = torch.concat((self.qpos_list[:1], val))

    def copy_frame(self):
        self.qpos_list[0] = self.qpos_list[self.substeps]

    def qstep(self, cur, action):
        torch_scale = self.simulator.get_torch_scale(action.device)
        substeps = self.substeps

        assert cur % substeps == 0
        action = action.reshape(-1, 6).clamp(-2., 2.) * torch_scale

        posrot = self.qpos_list[cur]
        pos = posrot[..., :3]
        rot = posrot[..., 3:7]

        pos, rot = rigid_body_motion((pos, rot),
            action[None,:].expand(substeps, -1, -1) * (
                torch.arange(substeps, device=action.device)[:, None, None]+1
                )/substeps)

        self.set_qpos(cur, torch.concat((pos, rot), axis=-1))
        
    def forward_kinematics(self, qpos):
        pos = qpos[..., :3]
        rot = qpos[..., 3:7]
        return pos, rot

    def get_state(self, index=0):
        return self.qpos_list[index].detach().cpu().numpy()

    def set_state(self, state, index=0):
        assert index == 0
        from tools.utils import totensor
        self.qpos_list = totensor(state, 'cuda:0')[None,:]

    @classmethod
    def empty_state(cls):
        #return np.zeros(self.qshape)
        qshape = cls.get_qshape()
        assert len(qshape) == (2,) and qshape[-1] == 7
        p = np.zeros(qshape)
        p[:, 3] = 1
        return p

    def add_tool_by_mode(self):
        if self._cfg.mode == 'Box':
            self.add_tool('Box', (self._cfg.size[0], self._cfg.size[1]/2 + self._cfg.size[0], self._cfg.size[2]))
        else:
            self.add_tool('Capsule', self._cfg.size)


class Gripper(Tool):
    def __init__(self, simulator, cfg=None, size=(0.02, 0.15, 0.02),
                 action_scale=(0.015, 0.015, 0.015, 0.05, 0.05, 0.05, 0.015), friction=10., mode='Box'):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool_by_mode()
        self.add_tool_by_mode()
        self._torch_scale = torch.tensor(
            np.array(self._cfg.action_scale),
            device='cuda:0', dtype=torch.float32
        )


    @classmethod
    def get_qshape(cls):
        return (3+3+1,) # center pos, y rotation, gap

    def get_action_shape(self):
        return (3+3+1,)

    @classmethod
    def empty_state(cls):
        p = np.zeros(7,)
        p[:3] = np.array([0.5, 0.15, 0.5])
        return p
    
    def qstep(self, cur, action):
        self.check_action(action)

        start_q = self.qpos_list[cur]

        pos = start_q[:3]
        rot = start_q[3:6]
        gap = start_q[-1]

        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None]+1)/self.substeps)
        pos = pos + action[None, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot + action[None, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi
        gap = gap + action[None, 6] * self._torch_scale[6] * substep_ratio
        gap = torch.relu(gap)  # gap must be greater than zero.
        self.set_qpos(cur, torch.cat([pos, rot, gap], -1))

    def forward_kinematics(self, qpos):
        #pos = torch.stack(pos - gap * ) 
        _rot = qpos[..., 3:6]
        gap = qpos[..., 6:7] * 0.1 + self._cfg.size[0] * 1.2

        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))

        dir = quaternion_apply(rot, torch.tensor([1., 0., 0], device=qpos.device, dtype=qpos.dtype))
        verticle = quaternion_apply(rot, torch.tensor([0., 1., 0], device=qpos.device, dtype=qpos.dtype))

        center = qpos[..., :3] + verticle * self._cfg.size[1] * 0.5

        left_pos = center - dir * gap
        right_pos = center + dir * gap
        pos_list = torch.stack((left_pos, right_pos), axis=-2)
        rot_list = torch.stack((rot, rot), axis=-2)
        return pos_list, rot_list

        

class Fingers(Gripper):
    # the action scale is controlled by lv..
    n_bodies = 2

    def __init__(self, simulator, cfg=None):
        super().__init__(simulator)


    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 6)

    @classmethod
    def get_action_shape(cls):
        return (cls.n_bodies * 6,)

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape)

    def qstep(self, cur, action):
        self.check_action(action)
        action = action.reshape(2, 6)

        start_q = self.qpos_list[cur].reshape(2, 6)
        pos = start_q[:, :3]
        rot = start_q[:, 3:6]
        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None, None]+1)/self.substeps)
        pos = pos[None, :] + action[None, :, :3] * self._torch_scale[:3] * substep_ratio
        rot = ((rot[None, :] + action[None, :, 3:6] * self._torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi
        self.set_qpos(cur, torch.cat([pos, rot], -1))

    def forward_kinematics(self, qpos):
        _rot = qpos[..., 3:6]
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(_rot, 'XYZ'))
        pos = qpos[..., :3].clone()
        # pos[..., 1] += self._cfg.size[1]
        return pos, rot


class Pusher(Tool):
    # the action scale is controlled by lv..
    n_bodies = 1

    def __init__(self, simulator, cfg=None, size=(0.03, 0.1, 0.1), lv=(0.012, 0.012, 0.012), friction=1., softness=666., K=0.0, mode='box'):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool_by_mode()

    @classmethod
    def get_qshape(cls): # state shape
        return (3,)

    @classmethod
    def get_action_shape(cls):
        return (3,)

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape) + np.array([0.5, 0.06, 0.3])

    def qstep(self, cur, action):
        torch_scale = self.simulator.get_torch_scale(action.device)
        substeps = self.substeps
        assert cur % substeps == 0
        action = action.reshape(-1, 3).clamp(-2., 2.) * torch_scale[..., :3]
        pos = self.qpos_list[cur]

        substep_ratio = ((torch.arange(self.substeps, device=action.device)+1)/self.substeps)
        pos = pos[None, :] + action * substep_ratio[:, None]

        self.set_qpos(cur, pos)
        
    def forward_kinematics(self, qpos):
        pos = qpos.clone().unsqueeze(-2)
        #if self._cfg.mode != 'box':
        #    pos[..., 1] += self._cfg.size[1]
        pos[..., 1] = torch.relu(pos[..., 1])
        rot = torch.zeros((*pos.shape[:-1], 4))
        rot[..., 0] = 1.
        return pos, rot


class PusherWithObstacle(Tool):
    # the action scale is controlled by lv..
    n_bodies = 2

    def __init__(self, simulator, cfg=None, 
        size=(0.03, 0.2), lv=(0.01, 0.01, 0.01), av=(0.01, 0.01, 0.01), 
        friction=40.
    ):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool('Box', 
            args=(
                self._cfg.size[0], 
                self._cfg.size[1]/2 + self._cfg.size[0], 
                self._cfg.size[1]/2 + self._cfg.size[0]
            )
        )
        self.add_tool('Capsule', (0.15, 0.0))

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 6)

    @classmethod
    def get_action_shape(cls):
        return ((cls.n_bodies - 1) * 6,)

    @classmethod
    def empty_state(cls):
        qshape = cls.get_qshape()
        return np.zeros(qshape)

    def qstep(self, cur, action):
        # print("qstep", action)
        torch_scale = self.simulator.get_torch_scale(action.device)
        substeps = self.substeps
        assert cur % substeps == 0
        action = torch.cat(
            [
                action, 
                torch.zeros(self.get_action_shape(), device='cuda', dtype=torch.float32)
            ], dim=0)
        action = action.reshape(-1, 2, 6).clamp(-2., 2.) * torch_scale
        pos = self.qpos_list[cur]

        substep_ratio = ((torch.arange(self.substeps, device=action.device)+1)/self.substeps)
        pos = pos[None, :] + action * substep_ratio[:, None, None]
        self.set_qpos(cur, pos)
        
    def forward_kinematics(self, qpos):
        # print("FK")
        euler_rot = qpos[..., 3:6] # (n, 3)
        # print(euler_rot)
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(euler_rot, 'XYZ'))
        pos_list, rot_list = qpos[..., :3], rot
        # print(euler_rot.shape)
        # print(pos_list.shape)
        # print(rot_list.shape)
        return pos_list, rot_list


class ContainerWithObstacle(Tool):

    n_bodies = 6
    size=(0.01, 0.09)

    def __init__(self, simulator, cfg=None, 
        size=(0.02, 0.2), lv=(0.01, 0.01, 0.01), av=(0.01, 0.01, 0.01), 
        friction=40.
    ):
        super().__init__(simulator)

    def init(self):
        Tool.init(self)
        self._cfg.size = self.size
        self.add_tool('Box', 
            args=(
                self._cfg.size[0], 
                (self._cfg.size[1]/2 + self._cfg.size[0]) / 2, 
                self._cfg.size[1]/2 + self._cfg.size[0]
            )
        )
        self.add_tool('Box', 
            args=(
                self._cfg.size[1]/2 + self._cfg.size[0], 
                self._cfg.size[0], 
                self._cfg.size[1]/2 + self._cfg.size[0]
            )
        )
        self.add_tool('Box', 
            args=(
                self._cfg.size[0], 
                (self._cfg.size[1]/2 + self._cfg.size[0]) / 2, 
                self._cfg.size[1]/2 + self._cfg.size[0]
            )
        )
        self.add_tool('Box', 
            args=(
                self._cfg.size[1]/2 + self._cfg.size[0], 
                (self._cfg.size[1]/2 + self._cfg.size[0]) / 2, 
                self._cfg.size[0], 
            )
        )
        self.add_tool('Box', 
            args=(
                self._cfg.size[1]/2 + self._cfg.size[0], 
                (self._cfg.size[1]/2 + self._cfg.size[0]) / 2, 
                self._cfg.size[0], 
            )
        )
        self.add_tool('Capsule', (0.17, 0.0))

    @classmethod
    def get_qshape(cls): # state shape
        return (cls.n_bodies, 6)

    @classmethod
    def get_action_shape(cls):
        return (3,)

    @classmethod
    def empty_state(cls):
        # qshape = cls.get_qshape()
        # return np.zeros(qshape)
        first = 0.8
        y_center = 0.12
        z_center = 0.5
        size = cls.size

        return np.array(
            [
                [first                  , y_center              , z_center              , 0, 0, 0], 
                [first - 1 * size[1] / 2, y_center - size[1] / 2, z_center              , 0, 0, 0], 
                [first - 2 * size[1] / 2, y_center              , z_center              , 0, 0, 0], 
                [first - 1 * size[1] / 2, y_center              , z_center - size[1] / 2, 0, 0, 0], 
                [first - 1 * size[1] / 2, y_center              , z_center + size[1] / 2, 0, 0, 0], 
                [0.5, -0.05, 0.5, 0, 0, 0]
            ]
        )

    def qstep(self, cur, action):
        # print("qstep", action)
        torch_scale = self.simulator.get_torch_scale(action.device)
        substeps = self.substeps
        assert cur % substeps == 0
        action = torch.cat(
            [
                # torch.stack([
                #     action[0], 
                #     torch.tensor(0, device='cuda', dtype=torch.float32), 
                #     action[1]], dim=-1), 
                action,
                torch.zeros(3, device='cuda', dtype=torch.float32)
            ], dim=0)
        action = torch.cat(
            [
                action, action, action, action, action, 
                torch.zeros(6, device='cuda', dtype=torch.float32)
            ], dim=0)
        
        action = action.reshape(-1, self.n_bodies, 6).clamp(-2., 2.) * torch_scale
        
        pos = self.qpos_list[cur]

        substep_ratio = ((torch.arange(self.substeps, device=action.device)+1)/self.substeps)
        pos = pos[None, :] + action * substep_ratio[:, None, None]
        self.set_qpos(cur, pos)
        
    def forward_kinematics(self, qpos):
        # print("FK")
        euler_rot = qpos[..., 3:6] # (n, 3)
        # print(euler_rot)
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(euler_rot, 'XYZ'))
        pos_list, rot_list = qpos[..., :3], rot
        return pos_list, rot_list


class RotatingPusher(Tool):

    n_bodies = 1
    
    def __init__(self, simulator, 
        cfg=None, friction=20., K=0., size=(0.03, 0.2), color=(0.3, 0.3, 0.3), 
        softness=666., lv=(0.01, 0.01, 0.01), av=(0.01, 0.01, 0.01)
    ):
        super().__init__(simulator)

    def init(self):
        super().init()
        self.add_tool('Box', 
            args=(
                self._cfg.size[0], 
                self._cfg.size[1]/2 + self._cfg.size[0], 
                self._cfg.size[1]/2 + self._cfg.size[0]
            )
        )

    @classmethod
    def get_qshape(cls):
        return (3+3,) # center pos, y rotation, gap

    def get_action_shape(self):
        return (3+3,)

    @classmethod
    def empty_state(cls):
        p = np.zeros(6,)
        p[:3] = np.array([0.5, 0.15, 0.5])
        return p

    def qstep(self, cur, action):
        torch_scale = self.simulator.get_torch_scale(action.device).squeeze()
        q = self.qpos_list[cur]
        pos, rot = q[:3], q[3:]

        # print(torch_scale.shape, pos.shape, rot.shape)
        substep_ratio = ((torch.arange(self.substeps, device=action.device)[:, None]+1)/self.substeps)
        pos = pos + action[None, :3] * torch_scale[:3] * substep_ratio
        rot = ((rot + action[None, 3:6] * torch_scale[3:6] * substep_ratio) + np.pi) % (2 * np.pi) - np.pi

        self.set_qpos(cur, torch.cat([pos, rot], dim=-1))
        # print(torch.cat([pos, rot], -1).shape)

    def forward_kinematics(self, qpos):
        # print("rotating forward_kinematics")
        euler_rot = qpos[..., 3:6] # (n, 3)
        from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
        rot = matrix_to_quaternion(euler_angles_to_matrix(euler_rot, 'XYZ'))

        pos_list, rot_list = qpos[..., :3].unsqueeze(-2), rot.unsqueeze(-2)

        
        # print(euler_rot.shape)
        # print(pos_list.shape)
        # print(rot_list.shape)
        return pos_list, rot_list


class Knife(Tool):
    def __init__(self, simulator, cfg=None):
        super().__init__(simulator)
        raise NotImplementedError

        
FACTORY = {
    'Gripper': Gripper,
    'Fingers': Fingers,
    'Pusher': Pusher,
    'PusherWithObstacle': PusherWithObstacle,
    'RotatingPusher': RotatingPusher,
    "ContainerWithObstacle": ContainerWithObstacle
}