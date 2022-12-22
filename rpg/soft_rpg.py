import tqdm
import torch
from .common_hooks import RLAlgo, build_hooks
from .env_base import GymVecEnv, TorchEnv
from .buffer import ReplayBuffer
from .info_net import InfoLearner
from .traj import Trajectory
from .intrinsic import IntrinsicMotivation
from .rnd import RNDExplorer
from .utils import create_hidden_space
from nn.distributions import Normal
from typing import Union
from tools.config import Configurable
from tools.utils import logger, totensor
from .policy_learner import PolicyLearner
from .hidden import HiddenSpace, Categorical
from .worldmodel import HiddenDynamicNet, DynamicsLearner
from . import repr 

class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,

        max_epoch=None,

        env_name=None,
        env_cfg=None,

        #z_dim=1, z_cont_dim=0,
        hidden=HiddenSpace.to_build(TYPE=Categorical),

        # model
        buffer=ReplayBuffer.dc,
        model=HiddenDynamicNet.dc,
        trainer=DynamicsLearner.dc,

        # policy
        pi_a=PolicyLearner.dc,
        pi_z=PolicyLearner.dc,

        head = Normal.gdc(
                    linear=False,
                    squash=True,
                    std_mode='statewise',
                    std_scale=1.,
                ),
        z_head=None,

        # mutual information term ..
        info = InfoLearner.dc,
        rnd = RNDExplorer.dc,

        # update parameters..
        horizon=6, batch_size=512, update_target_freq=2, update_train_step=1, warmup_steps=1000, steps_per_epoch=None,

        actor_delay=2, z_delay=0,
        eval_episode=10, save_video=0,

        # trainer utils ..
        hooks=None, path=None, wandb=None, log_date=False,
        tau=0.005, relabel=0.,

        time_embedding=0,

        seed=None,
    ):
        if seed is not None:
            from tools.utils import set_seed
            set_seed(seed)
    
        Configurable.__init__(self)
        RLAlgo.__init__(self, None, build_hooks(hooks))

        if env is None:
            from tools.config import CN
            from .env_base import make
            env = make(env_name, **CN(env_cfg))

        obs_space = env.observation_space
        #self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)
        #self.use_z = z_dim > 1 or z_cont_dim > 0
        z_space: HiddenSpace = HiddenSpace.build(hidden)
        self.z_space = z_space

        self.horizon = horizon
        self.env = env
        self.device = 'cuda:0'
        self.buffer = ReplayBuffer(obs_space, env.action_space, env.max_time_steps, horizon, cfg=buffer)
        self.dynamics_net = HiddenDynamicNet(obs_space, env.action_space, z_space, time_embedding, cfg=model).cuda()

        state_dim = self.dynamics_net.state_dim
        hidden_dim = 256

        from .policy_net import DiffPolicy, QPolicy
        pi_a_net = DiffPolicy(state_dim + z_space.dim, hidden_dim, Normal(env.action_space, cfg=head), time_embedding=time_embedding).cuda()
        self.pi_a = PolicyLearner('a', env.action_space, pi_a_net, z_space.tokenize, cfg=pi_a)

        # z learning ..
        z_head = self.z_space.make_policy_head(z_head)
        pi_z_net = QPolicy(state_dim, hidden_dim, z_head, time_embedding=time_embedding).cuda()
        self.pi_z = PolicyLearner('z', env.action_space, pi_z_net, z_space.tokenize, cfg=pi_z, ignore_hidden=True)

        #self.info_net = lambda x: x['rewards'], 0, 0
        self.model_learner = DynamicsLearner(self.dynamics_net, self.pi_a, self.pi_z, cfg=trainer)

        self.z = None
        self.update_step = 0

        self.intrinsics = [self.pi_a, self.pi_z]
        if self.z_space.learn:
            self.info_learner = InfoLearner(state_dim, env.action_space, z_space, cfg=info, hidden_dim=hidden_dim)
            self.intrinsics.append(self.info_learner)
        self.make_rnd()

        self.intrinsic_reward = IntrinsicMotivation(*self.intrinsics)
        self.model_learner.set_intrinsic(self.intrinsic_reward)

        z_space.callback(self)


    def make_rnd(self):
        rnd = self._cfg.rnd
        if rnd.scale > 0.:
            self.exploration = RNDExplorer(self.env.observation_space, self.dynamics_net.state_dim, self.buffer, self.dynamics_net.enc_s, cfg=rnd)

            if not self.exploration.as_reward:
                self.intrinsics.append(self.exploration)
        else:
            self.exploration = None

    def relabel_z(self, seg):
        z = seg.z
        # .obs_seq[0], seg.timesteps[0], seg.z
        if self._cfg.relabel > 0.:
            assert seg.future is not None
            o, no, a = seg.future
            traj = self.dynamics_net.enc_s(torch.stack((o, no)))
            new_z = self.info_learner.sample_z(traj, a)[0]
            # if self.update_step % 100 == 0:
            #     print(z)
            #     print(new_z)

            mask = torch.rand(size=(len(z),)) < self._cfg.relabel
            if new_z.dtype == torch.float32:
                #new_z = new_z.to(torch.int64)
                prior = torch.distributions.Normal(0, 1)
                mask[prior.log_prob(new_z).mean(axis=-1) < -2.] = False
                

            z = z.clone()
            z[mask] = new_z[mask]

            if new_z.dtype == torch.float32:
                logger.logkv_mean('relabel', new_z.std())
        return z

    def update_dynamcis(self):
        seg = self.buffer.sample(self._cfg.batch_size)
        
        if self.exploration is not None and self.exploration.as_reward:
            seg.reward = seg.reward + self.exploration.intrinsic_reward(seg.obs_seq[1:])

        z = self.relabel_z(seg)
        prev_z = z[None, :].expand(len(seg.obs_seq), *seg.z.shape)
        self.model_learner.learn_dynamics(
            seg.obs_seq, seg.timesteps, seg.action, seg.reward, seg.done, seg.truncated_mask, prev_z)
    
    def update_pi_a(self):
        if self.update_step % self._cfg.actor_delay == 0:
            seg = self.buffer.sample(self._cfg.batch_size, horizon=1)
            z = self.relabel_z(seg)
            rollout = self.dynamics_net.inference(
                seg.obs_seq[0], z, seg.timesteps[0], self.horizon, self.pi_z, self.pi_a,
                intrinsic_reward=self.intrinsic_reward
            )
            self.pi_a.update(rollout)
            if self.z_space.learn:
                self.info_learner.update(rollout)

            if self.exploration is not None:
                self.exploration.update_by_rollout(rollout)


            for k, v in rollout['extra_rewards'].items():
                logger.logkv_mean(f'reward_{k}', float(v.mean()))

    def update_pi_z(self):
        if self._cfg.z_delay > 0 and self.update_step % self._cfg.z_delay == 0:
            o, z, t = self.buffer.sample_start(self._cfg.batch_size)
            assert (t < 1).all()
            #z = self.relabel_z(o, t, z) relabel does not makes any sense here
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
        s = self.dynamics_net.enc_s(obs)
        z = self.pi_z(s, prevz, prev_action=prevz, timestep=timestep).a
        a = self.pi_a(s, z, timestep=timestep).a
        return a, z.detach().cpu().numpy()

    def inference(self, n_step, mode='training'):
        with torch.no_grad():
            z_space = self.z_space.space
            obs, timestep = self.start(self.env)
            if self.z is None:
                self.z = [z_space.sample()] * len(obs)
            for idx in range(len(obs)):
                if timestep[idx] == 0:
                    self.z[idx] = z_space.sample() * 0
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

                if mode != 'training' and self.exploration is not None:
                    # for visualize the transitions ..
                    transition['next_state'] = self.dynamics_net.enc_s(totensor(obs, device='cuda:0'))

                    if self.exploration is not None:
                        self.exploration.visualize_transition(transition)

                transitions.append(transition)
                timestep = transition['next_timestep']
                for j in range(len(obs)):
                    if timestep[j] == 0:
                        self.z[j] = z_space.sample() * 0

            if mode == 'training' and self.exploration is not None:
                self.exploration.add_data(obs)

            if self.buffer.total_size() > self._cfg.warmup_steps and self._cfg.update_train_step > 0 and mode == 'training':
                if idx % self._cfg.update_train_step == 0:
                    self.update()

        if len(images) > 0:
            logger.animate(images, 'eval.mp4')

        return Trajectory(transitions, len(obs), n_step)

    def setup_logger(self):
        format_strs = ["stdout", "log", "csv"]
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

        logger.configure(dir=self._cfg.path, format_strs=format_strs, date=self._cfg.log_date, **kwargs)

        import os
        with open(os.path.join(logger.get_dir(), 'config.yaml'), 'w') as f:
            f.write(str(self._cfg))

    def run_rpgm(self):
        self.setup_logger()
        env = self.env
        max_epoch = self._cfg.max_epoch

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
            epoch_id += 1

    def evaluate(self, env, steps):
        with torch.no_grad():
            self.start(self.env, reset=True)
            out = self.inference(steps * self._cfg.eval_episode, mode='evaluate')
            self.start(self.env, reset=True)
            return out