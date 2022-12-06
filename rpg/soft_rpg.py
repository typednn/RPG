import tqdm
import copy
import gym
import torch
from tools.config import Configurable
from tools.utils import RunningMeanStd, mlp, orthogonal_init, Seq, TimedSeq, logger, Identity, ema, CatNet, totensor
from tools.optim import LossOptimizer
from .common_hooks import RLAlgo, build_hooks
from typing import Union
from .env_base import GymVecEnv, TorchEnv
from torch.nn.functional import binary_cross_entropy as bce
from .utils import masked_temporal_mse, create_hidden_space
from .buffer import ReplayBuffer
from .info_net import InfoNet
from .traj import Trajectory
from .policy_learner import DiffPolicyLearner, DiscretePolicyLearner
from .intrinsic import IntrinsicMotivation


from .worldmodel import HiddenDynamicNet, DynamicsLearner
class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,

        z_dim=1, z_cont_dim=0,

        # model
        buffer=ReplayBuffer.dc,
        model=HiddenDynamicNet.dc,
        trainer=DynamicsLearner.dc,

        # policy
        pi_a=DiffPolicyLearner.dc,
        pi_z=DiscretePolicyLearner.dc,

        # mutual information term ..
        info = InfoNet.dc,

        # update parameters..
        horizon=6,
        batch_size=512,
        update_target_freq=2,
        update_train_step=1,
        warmup_steps=1000,
        steps_per_epoch=None,

        actor_delay=2,
        z_delay=0,
        eval_episode=10,

        # trainer utils ..
        hooks = None,
        path = None,
        wandb=None,
        save_video=0,

        tau=0.005,
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, None, build_hooks(hooks))

        obs_space = env.observation_space
        self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)

        self.horizon = horizon
        self.env = env
        self.device = 'cuda:0'
        self.buffer = ReplayBuffer(
            obs_space, env.action_space, env.max_time_steps, horizon, cfg=buffer)
        self.dynamics_net = HiddenDynamicNet(
            obs_space, env.action_space, z_space, cfg=model).cuda()

        state_dim = self.dynamics_net.state_dim
        enc_z = self.dynamics_net.enc_z
        hidden_dim = 256
        self.pi_a = DiffPolicyLearner(
            'a', state_dim, enc_z, hidden_dim, env.action_space, cfg=pi_a
        )
        self.pi_z = DiscretePolicyLearner(
            'z', state_dim, enc_z, hidden_dim, self.z_space, cfg=pi_z
        )
        #self.info_net = lambda x: x['rewards'], 0, 0
        self.intrinsic_reward = IntrinsicMotivation(self.pi_a, self.pi_z)
        self.model_learner = DynamicsLearner(self.dynamics_net, self.pi_a, self.pi_z, self.intrinsic_reward, cfg=trainer)

        self.z = None
        self.update_step = 0

    def update_dynamcis(self):
        seg = self.buffer.sample(self._cfg.batch_size)
        prev_z = seg.z[None, :].expand(len(seg.obs_seq), *seg.z.shape)
        self.model_learner.learn_dynamics(
            seg.obs_seq, seg.timesteps, seg.action, seg.reward, seg.done, seg.truncated_mask, prev_z)

    
    def update_pi_a(self):
        if self.update_step % self._cfg.actor_delay == 0:
            seg = self.buffer.sample(self._cfg.batch_size, horizon=1)
            rollout = self.dynamics_net.inference(
                seg.obs_seq[0], seg.z, seg.timesteps[0], self.horizon, self.pi_z, self.pi_a, intrinsic_reward=self.intrinsic_reward)
            self.pi_a.update(rollout)


    def update_pi_z(self):
        if self._cfg.z_delay > 0 and self.update_step % self._cfg.z_delay == 0:
            o, z, t = self.buffer.sample_start(self._cfg.batch_size)
            assert (t < 1).all()
            rollout = self.dynamics_net.inference(o, z, t, self.horizon, self.pi_z, self.pi_a, intrinsic_reward=self.intrinsic_reward)
            self.pi_z.update(rollout)

    def update(self):
        self.update_dynamcis()
        self.update_pi_a()
        self.update_pi_z()
        self.update_step += 1
        if self.update_step % self._cfg.update_target_freq == 0:
            self.model_learner.ema(self._cfg.tau)
            self.pi_a.ema(self._cfg.tau)
            self.pi_z.ema(self._cfg.tau)

    def eval(self):
        #print("Eval is not implemented")
        # eval not implemented
        return

    def train(self):
        # train
        return


    def policy(self, obs, prevz, timestep):
        obs = totensor(obs, self.device)
        prevz = totensor(prevz, self.device, dtype=None)
        timestep = totensor(timestep, self.device, dtype=None)
        s = self.dynamics_net.enc_s(obs, timestep=timestep)
        z = self.pi_z(s, prevz, prev_action=prevz, timestep=timestep).a
        a = self.pi_a(s, z).a
        return a, z.detach().cpu().numpy()


    def inference(self, n_step, mode='training'):
        with torch.no_grad():
            obs, timestep = self.start(self.env)
            if self.z is None:
                self.z = [self.z_space.sample()] * len(obs)
            for idx in range(len(obs)):
                if timestep[idx] == 0:
                    self.z[idx] = self.z_space.sample() * 0
        r = tqdm.trange if self._cfg.update_train_step > 0 else range 

        transitions = []
        images = []
        for idx in r(n_step):
            with torch.no_grad():
                self.eval()
                transition = dict(obs = obs, timestep=timestep)
                prevz = totensor(self.z, device=self.device, dtype=None)
                a, self.z = self.policy(obs, self.z, timestep)
                data, obs = self.step(self.env, a)

                if mode != 'training' and self._cfg.save_video > 0 and idx < self._cfg.save_video: # save video steps
                    images.append(self.env.render('rgb_array')[0])

                transition.update(**data, a=a, z=prevz)
                transitions.append(transition)
                timestep = transition['next_timestep']
                for j in range(len(obs)):
                    if timestep[j] == 0:
                        self.z[j] = self.z_space.sample() * 0

            if self.buffer.total_size() > self._cfg.warmup_steps and self._cfg.update_train_step > 0 and mode == 'training':
                if idx % self._cfg.update_train_step == 0:
                    self.update()

        if len(images) > 0:
            logger.animate(images, 'eval.mp4')

        return Trajectory(transitions, len(obs), n_step)

    def setup_logger(self):
        format_strs = ["stdout", "log", "csv", 'tensorboard']
        kwargs = {}
        if self._cfg.wandb is not None:
            wandb_cfg = dict(self._cfg.wandb)
            if 'stop' not in wandb_cfg:
                format_strs = format_strs[:3] + ['wandb']
                kwargs['config'] = self._cfg
                kwargs['project'] = wandb_cfg.get('project', 'mujoco_rl')
                kwargs['group'] = wandb_cfg.get('group', self.env.env_name)

                try:
                    name = self._cfg._exp_name
                except:
                    name = None
                kwargs['name'] = wandb_cfg.get('name', None) + (('_' + name) if name is not None else '')

        logger.configure(dir=self._cfg.path, format_strs=format_strs, **kwargs)

    def run_rpgm(self, max_epoch=None):
        self.setup_logger()
        env = self.env

        steps = self._cfg.steps_per_epoch or self.env.max_time_steps
        epoch_id = 0
        while True:
            if max_epoch is not None and epoch_id >= max_epoch:
                break

            traj = self.inference(steps)
            self.buffer.add(traj)

            a = traj.get_tensor('a')
            logger.logkv_mean('a_max', float(a.max()))
            logger.logkv_mean('a_min', float(a.min()))

            self.call_hooks(locals())
            print(traj.summarize_epsidoe_info())

            logger.dumpkvs()

    def evaluate(self, env, steps):
        with torch.no_grad():
            self.start(self.env, reset=True)
            out = self.inference(steps * self._cfg.eval_episode, mode='evaluate')
            self.start(self.env, reset=True)
            return out