import tqdm
import copy
import gym
import torch
from tools.config import Configurable
from tools.utils import RunningMeanStd, mlp, orthogonal_init, Seq, TimedSeq, logger, Identity, ema, CatNet, totensor
from tools.optim import LossOptimizer
from .common_hooks import RLAlgo, build_hooks
from .buffer import ReplayBuffer
from .traj import Trajectory
from typing import Union
from .env_base import GymVecEnv, TorchEnv
from nn.distributions import DistHead, NormalAction
from torch.nn.functional import binary_cross_entropy as bce
from .utils import masked_temporal_mse, create_hidden_space
from .worldmodel import GeneralizedQ

from .intrinsic import IntrinsicReward
from .soft_actor_critic import SoftQPolicy, PolicyA, SoftPolicyZ, ValuePolicy
from .intrinsic import EntropyLearner, InfoNet


class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,

        z_dim=1,
        z_cont_dim=0,

        # network parameters
        have_done=False,
        head=DistHead.to_build(
            TYPE='Normal',
            linear=False,
            std_mode='fix_no_grad',
            std_scale=0.2
        ),
        z_head=None,
        pi_z=SoftPolicyZ.dc,

        buffer = ReplayBuffer.dc,
        optim = LossOptimizer.gdc(lr=3e-4),
        dyna_optim=None,

        gamma=0.99, lmbda=0.97,
        weights=dict(state=1000., reward=0.5, q_value=0.5, done=1.),

        # mutual information term ..
        enta = EntropyLearner.dc,
        entz = EntropyLearner.gdc(coef=4., target_mode='none'),
        info = InfoNet.dc,

        # update parameters..
        horizon=6,
        batch_size=512,
        update_target_freq=2,
        update_train_step=1,
        warmup_steps=1000,
        steps_per_epoch=None,
        tau=0.005,
        z_delay=None,
        actor_delay=2,
        eval_episode=10,

        # trainer utils ..
        hooks = None,
        path = None,
        qmode='value',
        zero_done_value=True, # by default, let the value network to predict the done.
        state_layer_norm=False,
        state_batch_norm=False,
        no_state_encoder=False,

        worldmodel=GeneralizedQ.dc,
        wandb=None,

        dynamic_type='normal',
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, None, build_hooks(hooks))

        obs_space = env.observation_space
        #action_dim = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else 1
        self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)

        self.horizon = horizon
        self.env = env
        self.device = 'cuda:0'
        self.z_delay = z_delay or actor_delay
        assert actor_delay % self.z_delay == 0

        # buffer samples horizon + 1
        self.buffer = ReplayBuffer(obs_space.shape, env.action_space, env.max_time_steps, horizon, cfg=buffer)

        self.nets, self.intrinsic_reward = self.make_network(obs_space, env.action_space, z_space)
        with torch.no_grad():
            self.target_nets = copy.deepcopy(self.nets).cuda()
            self.target_nets.eval()

        self.nets.intrinsic_reward = self.intrinsic_reward
        self.target_nets.intrinsic_reward = self.intrinsic_reward

        self.actor_optim = LossOptimizer(self.nets.policies, cfg=optim)
        self.dyna_optim = LossOptimizer(self.nets.dynamics, cfg=optim, **(dyna_optim or {}))

        self.update_step = 0
        self.z = None

        self.sync_alpha()

    def sync_alpha(self):
        #self.nets.intrinsic_reward.sync_alpha()
        alpha_a = self.enta.alpha
        alpha_z = self.entz.alpha
        self.nets.set_alpha(alpha_a, alpha_z)
        self.target_nets.set_alpha(alpha_a, alpha_z)

    def update(self):
        self.nets.train()
        self.sync_alpha()
        obs_seq, timesteps, action, reward, done_gt, truncated_mask = self.buffer.sample(self._cfg.batch_size)

        # ---------------------- update dynamics ----------------------
        assert len(obs_seq) == len(timesteps) == len(action) + 1 == len(reward) + 1 == len(done_gt) + 1 == len(truncated_mask) + 1
        batch_size = len(obs_seq[0])

        with torch.no_grad():
            prev_z = self.intrinsic_reward.sample_posterior_z(self.nets.enc_s, obs_seq, action, timesteps)
            gt = dict(reward=reward)
            dyna_loss = dict()

            next_obs = obs_seq[1:].reshape(-1, *obs_seq.shape[2:])
            z_seq = prev_z[1:].reshape(-1, *prev_z.shape[2:])
            # zz = z_seq.reshape(-1).detach().cpu().numpy().tolist(); print([zz.count(i) for i in range(6)])
            next_timesteps = timesteps[1:].reshape(-1)

            samples = self.target_nets.inference(next_obs, z_seq, next_timesteps, self.horizon)
            qtarg = samples['value'].min(axis=-1)[0].reshape(-1, batch_size, 1)
            assert reward.shape == qtarg.shape == done_gt.shape, (reward.shape, qtarg.shape, done_gt.shape)

            if self._cfg.qmode == 'Q':
                gt['q_value'] = reward + (1-done_gt.float()) * self._cfg.gamma * qtarg
            else:
                if self._cfg.zero_done_value:
                    qtarg = qtarg * (1 - done_gt.float())
                gt['q_value'] = qtarg
                assert self._cfg.qmode == 'value'

            gt['state'] = samples['state'][0].reshape(-1, batch_size, samples['state'].shape[-1])
            logger.logkv_mean('q_value', float(gt['q_value'].mean()))
            logger.logkv_mean('reward_step_mean', float(reward.mean()))

        pred_traj = self.nets.inference(obs_seq[0], prev_z[0], timesteps[0], self.horizon, a_seq=action, z_seq=prev_z[1:]) 
        output = dict(state=pred_traj['state'][1:], q_value=pred_traj['q_value'],  reward=pred_traj['reward'])
        if self._cfg.qmode == 'value':
            output['q_value'] = pred_traj['pred_values']

        for k in ['state', 'q_value', 'reward']:
            dyna_loss[k] = masked_temporal_mse(output[k], gt[k], truncated_mask) / self.horizon

        # assert truncated_mask.all()
        if self._cfg.have_done:
            assert done_gt.shape == pred_traj['done'].shape
            loss = bce(pred_traj['done'], done_gt, reduction='none').mean(axis=-1) # predict done all the way ..
            dyna_loss['done'] = (loss * truncated_mask).sum(axis=0)
            logger.logkv_mean('done_acc', ((pred_traj['done'] > 0.5) == done_gt).float().mean())


        dyna_loss_total = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss]).mean(axis=0)
        self.dyna_optim.optimize(dyna_loss_total)
        info = {'dyna_' + k + '_loss': float(v.mean()) for k, v in dyna_loss.items()}
        logger.logkv_mean('dyna_total_loss', float(dyna_loss_total.mean()))

        s = output['state']
        logger.logkv_mean('state_mean', float(s.mean()))
        logger.logkv_mean('state_min', float(s.min()))
        logger.logkv_mean('state_max', float(s.max()))


        # ---------------------- update actor ----------------------
        rollout = self.nets.inference(obs_seq[0], prev_z[0], timesteps[0], self.horizon)

        if self.update_step % self.z_delay == 0:
            loss_z = self.nets.pi_z.loss(rollout)
            if self.update_step % self._cfg.actor_delay == 0:
                loss_a = self.nets.pi_a.loss(rollout)
                logger.logkv_mean('a_loss', float(loss_a))
            else:
                loss_a = 0.
            logger.logkv_mean('z_loss', float(loss_z))
            self.actor_optim.optimize(loss_a + loss_z)

        enta, entz = self.intrinsic_reward.get_ent_from_traj(rollout)
        logger.logkv_mean('a_ent', enta.mean())
        logger.logkv_mean('z_ent', entz.mean())
        if self.update_step % self.z_delay == 0:
            self.entz.update(entz)
            logger.logkv_mean('z_alpha', float(self.entz.alpha))
        if self.update_step % self._cfg.actor_delay == 0:
            self.enta.update(enta)
            logger.logkv_mean('a_alpha', float(self.enta.alpha))
        self.intrinsic_reward.update(rollout)


        self.update_step += 1
        if self.update_step % self._cfg.update_target_freq == 0:
            ema(self.nets, self.target_nets, self._cfg.tau)
            #ema(self.nets, self.target_nets, self._cfg.tau)
            self.intrinsic_reward.ema(self._cfg.tau)
        logger.logkvs_mean(info)


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
        for idx in r(n_step):
            with torch.no_grad():
                self.nets.eval()
                transition = dict(obs = obs, timestep=timestep)
                a, self.z = self.nets.policy(obs, self.z, timestep)
                # if mode != 'training' and idx == 0:
                #     print(self.z, self.nets.pi_z.q_value(self.nets.enc_s(obs, timestep=timestep)))
                data, obs = self.step(self.env, a)

                transition.update(**data, a=a, z=totensor(self.z, device=self.device, dtype=None))
                transitions.append(transition)
                timestep = transition['next_timestep']
                for idx in range(len(obs)):
                    if timestep[idx] == 0:
                        self.z[idx] = self.z_space.sample() * 0

            if self.buffer.total_size() > self._cfg.warmup_steps and self._cfg.update_train_step > 0 and mode == 'training':
                if idx % self._cfg.update_train_step == 0:
                    self.update()
        return Trajectory(transitions, len(obs), n_step)

    def run_rpgm(self, max_epoch=None):
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
        env = self.env
        self.sync_alpha()

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

    def make_dynamic_network(self, obs_space, action_space, z_space, hidden_dim, latent_dim):
        # TODO: layer norm?
        from .utils import ZTransform, config_hidden

        args = []
        class BN(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(dim)

            def forward(self, x):
                if len(x.shape) == 3:
                    return self.bn(x.transpose(1, 2)).transpose(1, 2)
                return self.bn(x)

        if self._cfg.state_layer_norm:
            args.append(torch.nn.LayerNorm(latent_dim, elementwise_affine=False))
        if self._cfg.state_batch_norm:
            args.append(BN(latent_dim))
        assert len(args) == 0

        if self._cfg.no_state_encoder:
            enc_s = TimedSeq(Identity(), *args) # encode state with time step ..
            latent_dim = obs_space.shape[0]
        else:
            enc_s = TimedSeq(mlp(obs_space.shape[0], hidden_dim, latent_dim), *args) # encode state with time step ..
        self.state_dim = latent_dim
        enc_z = ZTransform(z_space)

        layer = 1 # gru layers ..
        if self._cfg.dynamic_type == 'normal':
            if isinstance(action_space, gym.spaces.Box):
                enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)
            else:
                enc_a = torch.nn.Embedding(action_space.n, hidden_dim)
            a_dim = hidden_dim

            init_h = mlp(latent_dim, hidden_dim, hidden_dim)
            dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
            state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..
        elif self._cfg.dynamic_type == 'tiny':
            if isinstance(action_space, gym.spaces.Box):
                enc_a = Identity()
                a_dim = action_space.shape[0]
            else:
                enc_a = torch.nn.Embedding(action_space.n, hidden_dim)
                a_dim = action_space.n

            init_h = Identity()
            #dynamics = torch.nn.GRU(a_dim, latent_dim, layer)
            class Duplicate(torch.nn.Module):
                def forward(self, x):
                    return x, x
            dynamics = Seq(mlp(a_dim + latent_dim, hidden_dim, latent_dim), Duplicate())
            state_dec = Identity()
        else:
            raise NotImplementedError

        reward_predictor = Seq(mlp(a_dim + latent_dim, hidden_dim, 1)) # s, a predict reward .. 
        done_fn = mlp(a_dim + latent_dim, hidden_dim, 1) if self._cfg.have_done else None
        return enc_s, enc_z, enc_a, init_h, dynamics, reward_predictor, done_fn, state_dec

    def make_intrinsic_reward(self, obs_space, action_space, z_space, hidden_dim, latent_dim):
        self.enta = EntropyLearner(action_space, cfg=self._cfg.enta, device=self.device, lr=self._cfg.optim.lr)
        self.entz = EntropyLearner(z_space, cfg=self._cfg.entz, device=self.device, lr=self._cfg.optim.lr)
        if isinstance(action_space, gym.spaces.Box):
            self.info_net = InfoNet(latent_dim, action_space.shape[0], hidden_dim, z_space, cfg=self._cfg.info).cuda()
        else:
            self.info_net = None
        return IntrinsicReward(
            self.enta, self.entz, self.info_net, self._cfg.optim
        )

    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        state_dim = 100
        action_dim = action_space.shape[0]

        enc_s, enc_z, enc_a, init_h, dynamics, reward_predictor, done_fn, state_dec = self.make_dynamic_network(
            obs_space, action_space, z_space, hidden_dim, state_dim)
        state_dim = self.state_dim

        if self._cfg.qmode == 'Q':
            q_fn = SoftQPolicy(state_dim, action_dim, z_space, enc_z, hidden_dim)
        else:
            q_fn = ValuePolicy(state_dim, action_dim, z_space, enc_z, hidden_dim, zero_done_value=self._cfg.zero_done_value)

        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = PolicyA(state_dim, hidden_dim, enc_z, head)
        pi_z = SoftPolicyZ(state_dim, hidden_dim, enc_z, cfg=self._cfg.pi_z)

        network = GeneralizedQ(
            enc_s, enc_a, pi_a, pi_z,
            init_h, dynamics, state_dec, reward_predictor, q_fn,
            done_fn, None,
            cfg=self._cfg.worldmodel,
            gamma=self._cfg.gamma,
            lmbda=self._cfg.lmbda,
            horizon=self._cfg.horizon
        )
        network.apply(orthogonal_init)
        info_net = self.make_intrinsic_reward(
            obs_space, action_space, z_space, hidden_dim, state_dim
        )
        return network.cuda(), info_net