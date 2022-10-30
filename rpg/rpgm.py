# model-based verison RPG
import tqdm
import copy
import numpy as np
import torch
from tools.config import Configurable
from tools.utils import RunningMeanStd, mlp, orthogonal_init, Seq, TimedSeq, logger, Identity, ema, CatNet, totensor
from tools.optim import LossOptimizer
from .common_hooks import RLAlgo, build_hooks
from .buffer import ReplayBuffer
from .traj import Trajectory
from typing import Union
from .env_base import GymVecEnv, TorchEnv
from nn.distributions import DistHead
from torch.nn.functional import binary_cross_entropy as bce
from .utils import compute_value_prefix, masked_temporal_mse, create_hidden_space, config_hidden
from .worldmodel import GeneralizedQ


class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,

        z_dim=1,
        z_cont_dim=0,

        nsteps=2000,

        # network parameters
        have_done=False,
        head=DistHead.to_build(
            TYPE='Normal',
            linear=False,
            std_mode='fix_no_grad',
            std_scale=0.2
        ),
        z_head=None,
        buffer = ReplayBuffer.dc,
        obs_norm=False,

        actor_optim = LossOptimizer.gdc(lr=3e-4),
        dyna_optim=None,
        hooks = None,
        path = None,

        # update parameters..
        horizon=6,
        batch_size=512,
        update_target_freq=2,
        update_train_step=1,
        warmup_steps=1000,
        steps_per_epoch=None,
        tau=0.005,
        max_update_step=200,
        actor_delay=2,
        weights=dict(state=1000., prefix=0.5, value=0.5, done=1.),
        qnet=GeneralizedQ.dc,

        entropy_coef=0.,
        entropy_target=None,
        eval_episode=10,

        
        pg=False,
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, (RunningMeanStd(clip_max=10.) if obs_norm else None), build_hooks(hooks))

        obs_space = env.observation_space
        action_dim = env.action_space.shape[0]
        self.z_space = z_space = create_hidden_space(z_dim, z_cont_dim)

        self.horizon = horizon
        self.env = env
        self.device = 'cuda:0'

        # buffer samples horizon + 1
        self.buffer = ReplayBuffer(obs_space.shape, action_dim, env.max_time_steps, horizon, cfg=buffer)

        nets = self.nets = self.make_network(obs_space, env.action_space, z_space).cuda()
        with torch.no_grad():
            self.target_nets = copy.deepcopy(nets).cuda()

        # only optimize pi_a here
        kwargs = {}
        if dyna_optim is not None:
            kwargs = {**dyna_optim}
        self.actor_optim = LossOptimizer(nets.policies, cfg=actor_optim)
        self.dyna_optim = LossOptimizer(nets.dynamics, cfg=actor_optim, **kwargs)

        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, requires_grad=(entropy_target is not None), device=self.device))
        if entropy_target is not None:
            if entropy_target > 0:
                self._cfg.defrost()
                self._cfg.entropy_target = -action_dim
            self.entropy_optim = LossOptimizer(self.log_alpha, cfg=actor_optim, **kwargs) #TODO: change the optim ..
        self.update_step = 0

    def enc_s(self, obs, timestep):
        return self.nets.enc_s(obs, timestep=timestep)

    def get_posterior(self, states):
        from nn.distributions import DeterminisiticAction
        return DeterminisiticAction(totensor([self.z_space.sample()] * len(states), device=self.device, dtype=None))

    def sample_z(self, obs, timestep):
        return self.get_posterior(self.nets.enc_s(obs, timestep=timestep))

    def dynamic_loss(self, obs, action, next_obs, reward, done_gt, mask, alpha, timesteps):
        t = timesteps[0]
        pred_traj = self.nets.inference(obs, self.sample_z(obs, t).sample()[0], t, self.horizon, a_seq=action) # by default, just rollout for horizon steps ..

        next_timesteps = timesteps + 1
        with torch.no_grad():
            state_gt = self.target_nets.enc_s(next_obs, timestep=next_timesteps) # use the current model ..
            if 'value_prefix' in pred_traj:
                rewards = pred_traj['value_prefix']
                vprefix_gt = compute_value_prefix(reward, self.nets._cfg.gamma)
                raise NotImplementedError
            else:
                rewards = pred_traj['rewards']
                vprefix_gt = reward

        dyna_loss = {
            'state': masked_temporal_mse(pred_traj['states'], state_gt, mask),
            'prefix': masked_temporal_mse(rewards, vprefix_gt, mask)
        }
        if self._cfg.have_done:
            loss = bce(pred_traj['dones'], done_gt, reduction='none').mean(axis=-1) # predict done all the way ..
            dyna_loss['done'] = (loss * mask).sum(axis=0)
            logger.logkv_mean('done_acc', ((pred_traj['dones'] > 0.5) == done_gt).float().mean())

        batch_size = len(obs)
        with torch.no_grad():
            next_obs = next_obs.reshape(-1, *obs.shape[1:])
            next_timesteps = next_timesteps.reshape(-1)
            seq_z = self.sample_z(next_obs, next_timesteps).sample()[0]

            vtarg = self.target_nets.inference(next_obs, seq_z, next_timesteps,
                self.horizon, alpha=alpha, value_fn=self.target_nets.value_fn,
                # pi_a=self.target_nets.pi_a, pi_z=self.target_nets.pi_z
            )['value']

            done_mask = (1 - (done_gt.cumsum(0) > 0).float())
            vtarg = vtarg.min(axis=-1, keepdims=True)[0].reshape(self.horizon, batch_size, -1)
            vtarg = vtarg * done_mask #not sure if this will help ..

        dyna_loss['value'] = masked_temporal_mse(pred_traj['next_values'], vtarg, mask)

        return dyna_loss

    def update_actor(self, obs, alpha, timesteps):
        t = timesteps[0]
        init_z = self.sample_z(obs, t).sample()[0]
        samples = self.nets.inference(obs, init_z, t, self.horizon, alpha=alpha, pg=self._cfg.pg)
        value, entropy_term = samples['value'], samples['entropy_term']
        assert value.shape[-1] in [1, 2]
        if not self._cfg.pg:
            actor_loss = -value[..., 0].mean(axis=0)
        else:
            estimated_value = value[..., 0]
            baseline = self.nets.value(obs, init_z, timestep=t)[..., 0]
            logp_a = samples['logp_a'][0].sum(axis=-1)

            adv = (estimated_value -  baseline).detach()
            adv = (adv - adv.mean(axis=0))/(adv.std(axis=0) + 1e-8)
            # adv = estimated_value.detach()
            #assert adv.shape[-1] == logp_a.shape[-1]
            assert adv.shape == logp_a.shape
            actor_loss = - (logp_a * adv).mean(axis=0) - alpha * (-logp_a).mean(axis=0)
            # actor_loss = samples['logp_a'] * (-value[..., 0].detach())
        self.actor_optim.optimize(actor_loss)

        if self._cfg.entropy_target is not None:
            entropy_loss = -torch.mean(self.log_alpha.exp() * (self._cfg.entropy_target - entropy_term.detach()))
            self.entropy_optim.optimize(entropy_loss)

        return {
            'actor_loss': float(actor_loss),
            'entropy': float(entropy_term.mean()),
        }

    def update(self):
        obs, next_obs, action, reward, done_gt, truncated, timesteps = self.buffer.sample(self._cfg.batch_size)

        assert len(next_obs) == self.horizon
        with torch.no_grad():
            alpha = self._cfg.entropy_coef * torch.exp(self.log_alpha).detach()

        truncated_mask = torch.ones_like(truncated) # we weill not predict state after done ..
        truncated_mask[1:] = 1 - (truncated.cumsum(0)[:-1] > 0).float()
        mask = truncated_mask[..., 0] / self.horizon

        dyna_loss = self.dynamic_loss(obs, action, next_obs, reward, done_gt, mask, alpha, timesteps=timesteps)
        dyna_loss_total = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss]).mean(axis=0)
        self.dyna_optim.optimize(dyna_loss_total)

        info = {'dyna_' + k + '_loss': float(v.mean()) for k, v in dyna_loss.items()}
        info['a_alpha'] = float(alpha.mean())

        if self.update_step % self._cfg.actor_delay == 0:
            actor_updates = self.update_actor(obs, alpha, timesteps=timesteps)
            info.update(actor_updates)

        self.update_step += 1
        if self.update_step % self._cfg.update_target_freq == 0:
            ema(self.nets, self.target_nets, self._cfg.tau)
        logger.logkvs_mean(info)

    def sample_init_z(self, obs):
        return totensor(
            [self.z_space.sample() * 0] * len(obs),
            device=self.device, dtype=torch.long
        )

    def inference(self, n_step, mode='training'):
        with torch.no_grad():
            obs, timestep = self.start(self.env)
            transitions = []

        r = tqdm.trange if self._cfg.update_train_step > 0 else range 
        z = self.sample_init_z(obs)

        for idx in r(n_step):
            with torch.no_grad():
                transition = dict(obs = obs, timestep=timestep)
                pd, z = self.nets.policy(obs, z, timestep)

                scale = pd.dist.scale
                logger.logkvs_mean({'std_min': float(scale.min()), 'std_max': float(scale.max())})
                a, _ = pd.rsample()
                data, obs = self.step(self.env, a)

                transition.update(**data, a=a, z=z)
                transitions.append(transition)
                timestep = transition['next_timestep']

            if self.buffer.total_size() > self._cfg.warmup_steps and self._cfg.update_train_step > 0 and mode == 'training':
                if idx % self._cfg.update_train_step == 0:
                    self.update()

        return Trajectory(transitions, len(obs), n_step)

    def run_rpgm(self, max_epoch=None):
        logger.configure(dir=self._cfg.path, format_strs=["stdout", "log", "csv", 'tensorboard'])
        env = self.env

        steps = self._cfg.steps_per_epoch or self.env.max_time_steps
        epoch_id = 0
        while True:
            if max_epoch is not None and epoch_id >= max_epoch:
                break

            traj = self.inference(steps)
            self.buffer.add(traj)

            if self._cfg.update_train_step == 0:
                for _ in tqdm.trange(min(steps, self._cfg.max_update_step)):
                    self.update()

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

    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        latent_dim = 100

        z_dim = z_space.inp_shape[0]
        latent_z_dim = z_dim

        # TODO: layer norm?
        from .utils import ZTransform, config_hidden
        enc_s = TimedSeq(mlp(obs_space.shape[0], hidden_dim, latent_dim)) # encode state with time step ..
        enc_z = ZTransform(z_space)
        enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)

        layer = 1
        init_h = mlp(latent_dim, hidden_dim, hidden_dim)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
        # dynamics = MLPDynamics(hidden_dim)

        #value_prefix = Seq(mlp(hidden_dim, hidden_dim, 1))
        value_prefix = Seq(mlp(hidden_dim + latent_dim, hidden_dim, 1)) # s, a predict reward .. 
        done_fn = mlp(hidden_dim, hidden_dim, 1) if self._cfg.have_done else None
        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..


        v_in = latent_dim + latent_z_dim
        value = CatNet(
            Seq(mlp(v_in, hidden_dim, 1)),
            Seq(mlp(v_in, hidden_dim, 1)),
        ) #layer norm, if necesssary
        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = Seq(mlp(latent_dim + latent_z_dim, hidden_dim, head.get_input_dim()), head)

        # override it if we want to use different network 
        pi_z =  None #Seq(mlp(latent_dim + latent_z_dim, hidden_dim, zhead.get_input_dim()), zhead)

        network = GeneralizedQ(
            enc_s, enc_a, enc_z,
            pi_a, pi_z, init_h, dynamics, state_dec, value_prefix, value, done_fn, None,
            cfg=self._cfg.qnet,
            horizon=self._cfg.horizon)
        network.apply(orthogonal_init)
        return network