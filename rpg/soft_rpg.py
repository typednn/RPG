import tqdm
import copy
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
from .soft_actor_critic import SoftQPolicy, PolicyA, SoftPolicyZ
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
        actor_delay=2,
        eval_episode=10,

        # trainer utils ..
        hooks = None,
        path = None,
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, None, build_hooks(hooks))

        obs_space = env.observation_space
        action_dim = env.action_space.shape[0]
        self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)

        self.horizon = horizon
        self.env = env
        self.device = 'cuda:0'

        # buffer samples horizon + 1
        self.buffer = ReplayBuffer(obs_space.shape, action_dim, env.max_time_steps, horizon, cfg=buffer)

        self.nets, self.intrinsic_reward = self.make_network(obs_space, env.action_space, z_space)
        with torch.no_grad():
            self.target_nets = copy.deepcopy(self.nets).cuda()


        self.nets.intrinsic_reward = self.intrinsic_reward
        self.target_nets.intrinsic_reward = self.intrinsic_reward

        self.actor_optim = LossOptimizer(self.nets.policies, cfg=optim)
        self.dyna_optim = LossOptimizer(
            self.nets.dynamics, cfg=optim, **(dyna_optim or {}))

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
        self.sync_alpha()
        obs_seq, timesteps, action, reward, done_gt, truncated_mask = self.buffer.sample(self._cfg.batch_size)

        # ---------------------- update dynamics ----------------------
        assert len(obs_seq) == len(timesteps) == len(action) + 1 == len(reward) + 1 == len(done_gt) + 1 == len(truncated_mask) + 1
        batch_size = len(obs_seq[0])

        prev_z = self.intrinsic_reward.sample_posterior_z(self.nets.enc_s, obs_seq, timesteps)
        pred_traj = self.nets.inference(obs_seq[0], prev_z[0], timesteps[0], self.horizon, a_seq=action, z_seq=prev_z[1:]) 

        gt = dict(reward=reward)
        dyna_loss = dict()
        with torch.no_grad():
            next_obs = obs_seq[1:].reshape(-1, *obs_seq.shape[2:])
            z_seq = prev_z[1:].reshape(-1, *prev_z.shape[2:])
            next_timesteps = timesteps[1:].reshape(-1)

            samples = self.target_nets.inference(next_obs, z_seq, next_timesteps, self.horizon)
            qtarg = samples['value'].min(axis=-1)[0].reshape(-1, batch_size, 1)
            gt['q_value'] = reward + (1-done_gt.float()) * self._cfg.gamma * qtarg
            gt['state'] = samples['state'][0].reshape(-1, batch_size, samples['state'].shape[-1])


        output = dict(state=pred_traj['state'][1:], q_value=pred_traj['q_value'],  reward=pred_traj['reward'])
        for k in ['state', 'q_value', 'reward']:
            dyna_loss[k] = masked_temporal_mse(output[k], gt[k], truncated_mask) / self.horizon

        if self._cfg.have_done:
            loss = bce(pred_traj['done'], done_gt, reduction='none').mean(axis=-1) # predict done all the way ..
            dyna_loss['done'] = (loss * truncated_mask).sum(axis=0)
            logger.logkv_mean('done_acc', ((pred_traj['done'] > 0.5) == done_gt).float().mean())


        dyna_loss_total = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss]).mean(axis=0)
        self.dyna_optim.optimize(dyna_loss_total)
        info = {'dyna_' + k + '_loss': float(v.mean()) for k, v in dyna_loss.items()}


        # ---------------------- update actor ----------------------
        if self.update_step % self._cfg.actor_delay == 0:
            rollout = self.nets.inference(obs_seq[0], prev_z[0], timesteps[0], self.horizon)
            loss_a = self.nets.pi_a.loss(rollout)
            loss_b = self.nets.pi_z.loss(rollout)
            self.actor_optim.optimize(loss_a + loss_b)

            self.intrinsic_reward.update(rollout)

        self.update_step += 1
        if self.update_step % self._cfg.update_target_freq == 0:
            ema(self.nets, self.target_nets, self._cfg.tau)
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
                transition = dict(obs = obs, timestep=timestep)
                a, self.z = self.nets.policy(obs, self.z, timestep)
                data, obs = self.step(self.env, a)

                transition.update(**data, a=a, z=totensor(self.z, device=self.device))
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
        logger.configure(dir=self._cfg.path, format_strs=["stdout", "log", "csv", 'tensorboard'])
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
        hidden_dim = 256
        latent_dim = 100
        # TODO: layer norm?
        from .utils import ZTransform, config_hidden
        enc_s = TimedSeq(mlp(obs_space.shape[0], hidden_dim, latent_dim)) # encode state with time step ..
        enc_z = ZTransform(z_space)
        enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)

        layer = 1
        init_h = mlp(latent_dim, hidden_dim, hidden_dim)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)

        reward_predictor = Seq(mlp(hidden_dim + latent_dim, hidden_dim, 1)) # s, a predict reward .. 
        done_fn = mlp(hidden_dim, hidden_dim, 1) if self._cfg.have_done else None
        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..
        return enc_s, enc_z, enc_a, init_h, dynamics, reward_predictor, done_fn, state_dec

    def make_intrinsic_reward(self, obs_space, action_space, z_space, hidden_dim, latent_dim):
        self.enta = EntropyLearner(action_space, cfg=self._cfg.enta, device=self.device)
        self.entz = EntropyLearner(z_space, cfg=self._cfg.entz, device=self.device)
        self.info_net = InfoNet(latent_dim, action_space.shape[0], hidden_dim, z_space, cfg=self._cfg.info).cuda()
        return IntrinsicReward(
            self.enta, self.entz, self.info_net, self._cfg.optim
        )

    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        state_dim = 100
        action_dim = action_space.shape[0]

        enc_s, enc_z, enc_a, init_h, dynamics, reward_predictor, done_fn, state_dec = self.make_dynamic_network(
            obs_space, action_space, z_space, hidden_dim, state_dim)


        q_fn = SoftQPolicy(state_dim, action_dim, z_space, hidden_dim)
        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = PolicyA(state_dim, hidden_dim, enc_z, head)
        pi_z = SoftPolicyZ(state_dim, hidden_dim, enc_z, cfg=self._cfg.pi_z)

        network = GeneralizedQ(
            enc_s, enc_a, pi_a, pi_z,
            init_h, dynamics, state_dec, reward_predictor, q_fn,
            done_fn, None,
            gamma=self._cfg.gamma,
            lmbda=self._cfg.lmbda,
            horizon=self._cfg.horizon
        )

        network.apply(orthogonal_init)

        info_net = self.make_intrinsic_reward(
            obs_space, action_space, z_space, hidden_dim, state_dim
        )
        return network.cuda(), info_net