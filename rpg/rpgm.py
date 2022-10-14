# model-based verison RPG
import copy
import torch
from tools.config import Configurable
from tools.nn_base import Network
from tools.utils import RunningMeanStd, mlp, orthogonal_init, Seq, logger, totensor, Identity, ema
from tools.optim import LossOptimizer
from .utils import compute_gae_by_hand
from .common_hooks import RLAlgo, build_hooks
from .buffer import ReplayBuffer
from .traj import Trajectory
from typing import Union
from .env_base import GymVecEnv, TorchEnv
from nn.distributions import DistHead


def compute_value_prefix(rewards, gamma):
    value_prefix = [None] * len(rewards)
    v = 0
    for i in range(len(rewards) - 1, -1, 0):
        v = v * gamma + rewards[i]
        value_prefix.append(v)
    return torch.stack(value_prefix)


class GeneralizedQ(Network):
    def __init__(
        self,
        enc_s, enc_a, enc_z,
        pi_a, pi_z,

        init_h, dynamic_fn,
        state_dec, value_prefix, value_fn,

        cfg=None,
        gamma=0.995,
        lmbda=0.97,
    ) -> None:
        super().__init__()

        self.pi_a = pi_a
        self.pi_z = pi_z
        assert self.pi_z is None


        self.enc_s = enc_s
        self.enc_z = enc_z
        self.enc_a = enc_a
        self.init_h = init_h
        self.dynamic_fn: torch.nn.GRU = dynamic_fn

        self.state_dec = state_dec
        self.value_prefix = value_prefix
        self.value_fn = value_fn
            
        def get_dynamics():
            return [enc_s, enc_a, enc_z, init_h, dynamic_fn, state_dec, value_prefix, value_fn]
        self.dynamics = torch.nn.ModuleList(get_dynamics())


    def policy(self, obs, z):
        return self.pi_a(self.enc_s(totensor(obs, self.device)), self.enc_z(z))

    def inference(self, obs, z, step, a_seq=None):
        z_embed = self.enc_z(z)
        assert z_embed is None
        s = self.enc_s(obs)
        h = self.init_h(s)[None, ] # GRU of layer 1

        hidden, logp_a, actions, states = [], [], [], [self.state_dec(h)]
        for idx in range(step):
            assert self.pi_z is None, "this means that we will not update z, anyway.."

            if a is None:
                a, logp = self.pi_a(s, z_embed).rsample()
            else:
                a, logp = a_seq[idx], torch.zeros((len(obs),), device=self.device)

            o, h = self.dynamic_fn(a, h)
            o = o[-1]
            assert o.shape[0] == a.shape[0] 
            s = self.state_dec(o) # predict the next hidden state ..

            hidden.append(o)
            actions.append(a)
            logp_a.append(logp)
            states.append(s)
        ss = torch.stack
        return dict(hidden=ss(hidden), actions=ss(actions), logp_a=ss(logp_a), states=ss(states))

    def predict_values(self, hidden, z_embed, rewards=None):
        hidden = torch.stack(hidden)
        value_prefix = self.value_prefix(hidden) # predict reward

        if rewards is not None:
            value_prefix = value_prefix + compute_value_prefix(rewards)

        values = self.value_fn(hidden, z_embed) # predict V(s, z_{t-1})

        gamma = 1
        lmbda = 1
        sum_lmbda = 0.
        v = 0
        for i in range(len(hidden)):
            vpred = (value_prefix[i] + values[i] * gamma * self._cfg.gamma)
            v = v + vpred * lmbda
            sum_lmbda += lmbda
            gamma *= self._cfg.gamma
            lmbda *= self._cfg.lmbda
        v = (v + (1./(1-self._cfg.lmbda) - sum_lmbda) * vpred) * (1-self._cfg.lmbda)
        return torch.cat((v[None,:], values)), value_prefix

    def value(self, obs, z, horizon, alpha=0):
        traj = self.inference(obs, z, horizon)
        extra_reward = traj['logp_a'] * alpha if alpha > 0 else 0
        return self.predict_values(traj['hidden'], traj['z_embed'], extra_reward) 


class train_model_based(RLAlgo, Configurable):
    def __init__(
        self, env: Union[GymVecEnv, TorchEnv],
        cfg=None,
        nsteps=2000,
        head=DistHead.to_build(TYPE='Normal', std_mode='fix_learnable', std_scale=0.5),
        horizon=6,
        buffer = ReplayBuffer.dc,
        obs_norm=True,

        actor_optim = LossOptimizer.gdc(lr=3e-4),
        hooks = None,
        path = None,

        batch_size=512,
        update_freq=2, # update target network ..
        update_step=100,
        tau=0.01,
        rho=0.7, # horizon decay
        weights=dict(state=2., prefix=0.5, value=0.5)
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, RunningMeanStd(clip_max=10.) if obs_norm else None, build_hooks(hooks))

        obs_space = env.observation_space,
        action_dim = env.action_space.shape[0]

        self.horizon = horizon
        self.env = env

        # buffer samples horizon + 1
        self.buffer = ReplayBuffer(obs_space, action_dim, env.max_time_steps, horizon)
        self.nets = self.make_network(env.observation_space, env.action_space).cuda()

        with torch.no_grad():
            self.target_nets = copy.deepcopy(self.nets).cuda()

        # only optimize pi_a here
        self.actor_optim = LossOptimizer(self.nets.pi_a.parameters(), cfg=actor_optim)
        self.dyna_optim = LossOptimizer(self.nets.dynamics.parameters(), cfg=actor_optim)


    def make_network(self, obs_space, action_dim, z_space=None):
        hidden_dim = 256
        latent_dim = 100 # encoding of state ..
        enc_s = mlp(obs_space.shape[0], hidden_dim, latent_dim) # TODO: layer norm?
        enc_z = Identity()
        enc_a = mlp(action_dim, hidden_dim, hidden_dim)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, 1) # num layer 1..

        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..
        init_h = mlp(latent_dim, hidden_dim, hidden_dim) # reconstruct obs ..

        head = DistHead.build((action_dim,), cfg=self._cfg.head)
        pi_a = Seq(mlp(hidden_dim, hidden_dim, head.get_input_dim()), head)
        value = Seq(mlp(hidden_dim, hidden_dim, 1))#layer norm, if necesssary
        value_prefix = Seq(mlp(hidden_dim, hidden_dim, 1))
        network = GeneralizedQ(enc_s, enc_a, enc_z, pi_a, None, init_h, dynamics, state_dec,  value_prefix, value)
        network.apply(orthogonal_init)
        return network
        
    def update(self, data, update_target):
        obs, next_obs, action, reward, idxs, weights = data

        with torch.no_grad():
            vtarg = self.target_nets.value(obs.reshape(-1, *obs.shape[2:]), None,
                self.horizon, alpha=0.).reshape(obs.shape[0], -1, 1)

            state_gt = self.nets.enc_s(torch.cat((obs, next_obs[-1:])))
            vprefix_gt = compute_value_prefix(reward, self._cfg.gamma)

        # update the dynamics model ..
        #self.nets.
        traj = self.nets.inference(obs, None, self.horizon+1, action)
        states = traj['states']
        values, value_prefix = self.nets.predict_values(traj['hidden'], None, None)

        horizon_weights = (self._cfg.rho ** torch.arange(self.horizon+1, device=obs.device))[:, None]
        horizon_sum = horizon_weights.sum()
        def hmse(a, b):
            h = horizon_weights[:len(a)]
            return (((a-b)**2).mean(axis=-1) * h).sum(axis=0)/horizon_sum

        dyna_loss = {'state': hmse(states, state_gt), 'value': hmse(values, vtarg), 'prefix': hmse(value_prefix, vprefix_gt)}
        loss = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss])
        assert loss.shape == weights.shape
        self.dyna_optim.optimize((loss * weights).sum(axis=0))

        logger.logkvs_mean(dyna_loss, prefix='dyna/')

        # update the value network ..
        value = self.nets.value(obs, alpha=0.)[0][0]
        self.actor_optim.optimize(-value.mean(axis=0))

        if update_target:
            ema(self.nets, self.target_nets, self._cfg.tau)

        return loss.detach().cpu().numpy()



    def rollout(self, n_step):
        obs, timestep = self.start(self.env)
        assert (timestep == 0).all()
        transitions = []
        for _ in range(n_step):
            transition = dict(obs = obs)
            a, _ = self.nets.policy(obs, None).rsample()
            data, obs = self.step(self.env, a)
            transition.update(**data, a)
            transitions.append(transition)
        return Trajectory(transitions, len(obs), n_step)


    def run_rpgm(self, max_epoch=None):

        n_step = self.env.max_time_steps
        logger.configure(dir=self._cfg.path)

        epoch_id = 0
        while True:
            if max_epoch is not None and epoch_id >= max_epoch:
                break
            traj = self.rollout(n_step)
            self.buffer.add(traj)
            for i in range(min(n_step, self._cfg.update_step)):
                self.update(self.buffer.sample(self._cfg.batch_size))