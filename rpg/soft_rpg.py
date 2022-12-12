import tqdm
import torch
from .common_hooks import RLAlgo, build_hooks
from .env_base import GymVecEnv, TorchEnv
from .buffer import ReplayBuffer
from .info_net import InfoLearner
from .traj import Trajectory
from .policy_learner import DiffPolicyLearner, DiscretePolicyLearner
from .intrinsic import IntrinsicMotivation
from .rnd import RNDOptim
from .utils import create_hidden_space
from typing import Union
from tools.config import Configurable
from tools.utils import logger, totensor
#from tools.optim import LossOptimizer
#from torch.nn.functional import binary_cross_entropy as bce


from .worldmodel import HiddenDynamicNet, DynamicsLearner
class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,

        env_name=None,
        env_cfg=None,

        z_dim=1, z_cont_dim=0,

        # model
        buffer=ReplayBuffer.dc,
        model=HiddenDynamicNet.dc,
        trainer=DynamicsLearner.dc,

        # policy
        pi_a=DiffPolicyLearner.dc,
        pi_z=DiscretePolicyLearner.dc,

        # mutual information term ..
        info = InfoLearner.dc,
        rnd = RNDOptim.dc,

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

        relabel=0.,
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, None, build_hooks(hooks))

        if env is None:
            from tools.config import CN
            from .env_base import make
            env = make(env_name, **CN(env_cfg))

        obs_space = env.observation_space
        self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)
        self.use_z = z_dim > 1 or z_cont_dim > 0

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

        if self.use_z:
            self.info_learner = InfoLearner(state_dim, env.action_space, hidden_dim, z_space, cfg=info)

        self.pi_a = DiffPolicyLearner(
            'a', state_dim, enc_z, hidden_dim, env.action_space, cfg=pi_a
        )
        self.pi_z = DiscretePolicyLearner(
            'z', state_dim, enc_z, hidden_dim, self.z_space, cfg=pi_z
        )
        self.intrinsics = [self.pi_a, self.pi_z]
        #self.info_net = lambda x: x['rewards'], 0, 0
        self.model_learner = DynamicsLearner(self.dynamics_net, self.pi_a, self.pi_z, cfg=trainer)

        self.z = None
        self.update_step = 0

        self.make_rnd()


    def make_rnd(self):
        rnd = self._cfg.rnd
        if rnd.rnd_scale > 0.:
            self.rnd = RNDOptim(self.env.observation_space, self.dynamics_net.state_dim, cfg=rnd)
            self.intrinsics.append(self.rnd)


    def get_intrinsic(self):
        if self.use_z:
            self.intrinsics.append(self.info_learner)
        return self.intrinsics

    def relabel_z(self, state, timestep, z):
        if self._cfg.relabel > 0.:
            #TODO: other relabel method, use the future state's info to relabel the current state's z
            new_z = self.info_learner.sample_z(self.dynamics_net.enc_s(state, timestep=timestep))[0]

            mask = torch.rand(size=(len(state),)) < self._cfg.relabel
            z = z.clone()
            z[mask] = new_z[mask]
        return z

    def update_dynamcis(self):
        seg = self.buffer.sample(self._cfg.batch_size)
        z = self.relabel_z(seg.obs_seq[0], seg.timesteps[0], seg.z)
        prev_z = z[None, :].expand(len(seg.obs_seq), *seg.z.shape)
        self.model_learner.learn_dynamics(
            seg.obs_seq, seg.timesteps, seg.action, seg.reward, seg.done, seg.truncated_mask, prev_z)
        self.intrinsic_reward.update_with_buffer(seg)

    
    def update_pi_a(self):
        if self.update_step % self._cfg.actor_delay == 0:
            seg = self.buffer.sample(self._cfg.batch_size, horizon=1)
            z = self.relabel_z(seg.obs_seq[0], seg.timesteps[0], seg.z)
            rollout = self.dynamics_net.inference(
                seg.obs_seq[0], z, seg.timesteps[0], self.horizon, self.pi_z, self.pi_a,
                intrinsic_reward=self.intrinsic_reward
            )
            self.pi_a.update(rollout)

            # update intrinsic reward
            self.intrinsic_reward.update_with_rollout(rollout)

            for k, v in rollout['extra_rewards'].items():
                logger.logkv_mean(f'reward_{k}', float(v.mean()))


    def update_pi_z(self):
        if self._cfg.z_delay > 0 and self.update_step % self._cfg.z_delay == 0:
            o, z, t = self.buffer.sample_start(self._cfg.batch_size)
            assert (t < 1).all()
            z = self.relabel_z(o, t, z)
            rollout = self.dynamics_net.inference(o, z, t, self.horizon, self.pi_z, self.pi_a, intrinsic_reward=self.intrinsic_reward)
            self.pi_z.update(rollout)

    def update(self):
        self.update_dynamcis()
        self.update_pi_a()
        # self.update_pi_z()
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
                a, self.z = self.policy(obs, self.z, timestep)
                prevz = totensor(self.z, device=self.device, dtype=None)
                data, obs = self.step(self.env, a)

                if mode != 'training' and self._cfg.save_video > 0 and idx < self._cfg.save_video: # save video steps
                    images.append(self.env.render('rgb_array')[0])

                transition.update(**data, a=a, z=prevz)

                if mode != 'training':
                    # for visualize the transitions ..
                    transition['next_state'] = self.dynamics_net.enc_s(obs, timestep=timestep)
                    self.intrinsic_reward.visualize_transition(transition)

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
        self.intrinsic_reward = IntrinsicMotivation(*self.get_intrinsic())
        self.model_learner.set_intrinsic(self.intrinsic_reward)

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

            