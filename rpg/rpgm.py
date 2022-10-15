# model-based verison RPG
import tqdm
import copy
import torch
from tools.config import Configurable
from tools.nn_base import Network
from tools.utils import RunningMeanStd, mlp, orthogonal_init, Seq, logger, totensor, Identity, ema, CatNet
from tools.optim import LossOptimizer
from .common_hooks import RLAlgo, build_hooks
from .buffer import ReplayBuffer
from .traj import Trajectory
from typing import Union
from .env_base import GymVecEnv, TorchEnv
from nn.distributions import DistHead

def compute_value_prefix(rewards, gamma):
    v = 0
    discount = 1
    value_prefix = []
    for i in range(len(rewards)):
        v = v + rewards[i] * discount
        value_prefix.append(v)
        discount = discount * gamma
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
        predict_q=False,
    ) -> None:
        super().__init__()

        self.pi_a, self.pi_z = pi_a, pi_z
        self.enc_s, self.enc_z, self.enc_a = enc_s, enc_z, enc_a
        assert self.pi_z is None

        self.init_h = init_h
        self.dynamic_fn: torch.nn.GRU = dynamic_fn

        self.state_dec = state_dec
        self.value_prefix = value_prefix
        self.value_fn = value_fn
            
        def get_dynamics():
            return [enc_s, enc_a, enc_z, init_h, dynamic_fn, state_dec, value_prefix, value_fn]
        self.dynamics = torch.nn.ModuleList(get_dynamics())


    def policy(self, obs, z):
        assert z is None
        obs = totensor(obs, self.device)
        s = self.enc_s(obs)
        return self.pi_a(s, self.enc_z(z))

    def inference(self, obs, z, step, a_seq=None):
        """
        s      a_1    a_2    ...
        |      |      |
        h_0 -> h_1 -> h_2 -> h_3 -> h_4
               |      |      |      |   
               o_1    o_2    o_3    o_4
             / |  \
            s1 r1  v1    
        """
        z_embed = self.enc_z(z)
        assert z_embed is None
        s = self.enc_s(obs)
        h = self.init_h(s)[None, ] # GRU of layer 1

        hidden, logp_a, actions, states = [], [], [], []
        for idx in range(step):
            assert self.pi_z is None, "this means that we will not update z, anyway.."

            if a_seq is None:
                a, logp = self.pi_a(s, z_embed).rsample()
            else:
                a, logp = a_seq[idx], torch.zeros((len(obs),), device=self.device)

            o, h = self.dynamic_fn(self.enc_a(a)[None, :], h)
            o = o[-1]
            assert o.shape[0] == a.shape[0] 
            s = self.state_dec(o) # predict the next hidden state ..

            hidden.append(o)
            actions.append(a)
            logp_a.append(logp)
            states.append(s)

        ss = torch.stack
        return dict(hidden=ss(hidden), actions=ss(actions), logp_a=ss(logp_a), states=ss(states), z_embed=None)

    def predict_values(self, hidden, z_embed, rewards=None):
        value_prefix = self.value_prefix(hidden) # predict reward prefix

        if self._cfg.predict_q:
            prefix = torch.zeros_like(value_prefix)
        else:
            prefix = value_prefix

        if rewards is not None:
            prefix = prefix + compute_value_prefix(rewards, self._cfg.gamma)

        values = self.value_fn(hidden, z_embed) # predict V(s, z_{t-1})

        gamma = 1
        lmbda = 1
        sum_lmbda = 0.
        v = 0
        for i in range(len(hidden)):
            if self._cfg.predict_q:
                vpred = values[i] + prefix[i] # add the additional rewards like entropies 
            else:
                vpred = (prefix[i] + values[i] * gamma * self._cfg.gamma)

            v = v + vpred * lmbda
            sum_lmbda += lmbda
            gamma *= self._cfg.gamma
            lmbda *= self._cfg.lmbda
        v = (v + (1./(1-self._cfg.lmbda) - sum_lmbda) * vpred) * (1-self._cfg.lmbda)

        return v, values, value_prefix

    def value(self, obs, z, horizon, alpha=0, action_penalty=0.):
        traj = self.inference(obs, z, horizon)
        extra_reward = traj['logp_a'] * alpha if alpha > 0 else None
        if action_penalty > 0.:
            penalty = action_penalty * (1 - traj['actions']**2).mean(axis=-1, keepdims=True)
            if extra_reward is None:
                extra_reward = penalty
            else: 
                extra_reward += penalty
        return self.predict_values(traj['hidden'], traj['z_embed'], extra_reward)


class Trainer(Configurable, RLAlgo):
    def __init__(
        self,
        env: Union[GymVecEnv, TorchEnv],
        cfg=None,
        nsteps=2000,
        head=DistHead.to_build(TYPE='Normal', linear=False, std_mode='fix_no_grad', std_scale=0.2),
        horizon=6,
        supervised_horizon=None,
        buffer = ReplayBuffer.dc,
        obs_norm=False,

        actor_optim = LossOptimizer.gdc(lr=3e-4),
        hooks = None,
        path = None,

        batch_size=512,
        update_freq=2, # update target network ..
        update_step=200,
        tau=0.005,
        rho=0.7, # horizon decay
        weights=dict(state=2., prefix=0.5, value=0.5),
        qnet=GeneralizedQ.dc,
        action_penalty=0.
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, (RunningMeanStd(clip_max=10.) if obs_norm else None), build_hooks(hooks))

        obs_space = env.observation_space
        action_dim = env.action_space.shape[0]

        self.horizon = horizon
        self.env = env

        # buffer samples horizon + 1
        self.buffer = ReplayBuffer(obs_space.shape, action_dim, env.max_time_steps, horizon, cfg=buffer)
        self.nets = self.make_network(obs_space, env.action_space).cuda()

        with torch.no_grad():
            self.target_nets = copy.deepcopy(self.nets).cuda()

        # only optimize pi_a here
        self.actor_optim = LossOptimizer(self.nets.pi_a, cfg=actor_optim)
        self.dyna_optim = LossOptimizer(self.nets.dynamics, cfg=actor_optim)

    def evaluate(self, env, steps):
        with torch.no_grad():
            return self.inference(steps)

    def make_network(self, obs_space, action_space):
        hidden_dim = 256
        latent_dim = 100 # encoding of state ..
        enc_s = mlp(obs_space.shape[0], hidden_dim, latent_dim) # TODO: layer norm?
        enc_z = Identity()
        enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)

        init_h = mlp(latent_dim, hidden_dim, hidden_dim) # reconstruct obs ..
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, 1) # num layer 1..
        value = Seq(mlp(hidden_dim, hidden_dim, 1))#layer norm, if necesssary
        value_prefix = Seq(mlp(hidden_dim, hidden_dim, 1))
        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..

        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = Seq(mlp(latent_dim, hidden_dim, head.get_input_dim()), head)
        network = GeneralizedQ(enc_s, enc_a, enc_z, pi_a, None, init_h, dynamics, state_dec,  value_prefix, value, cfg=self._cfg.qnet)
        network.apply(orthogonal_init)
        return network
        
    def update(self, buffer: ReplayBuffer, update_target):
        supervised_horizon = self.horizon # not +1 yet

        obs, next_obs, action, reward, idxs, weights = buffer.sample(self._cfg.batch_size)
        batch_size = obs.shape[0]

        with torch.no_grad():
            vtarg = self.target_nets.value(next_obs.reshape(-1, *obs.shape[1:]), None,
                self.horizon, alpha=0.)[0].reshape(next_obs.shape[0], batch_size, -1)[:supervised_horizon]
            state_gt = self.nets.enc_s(next_obs)[:supervised_horizon]
            vprefix_gt = compute_value_prefix(reward, self.nets._cfg.gamma)[:supervised_horizon]
            assert torch.allclose(reward[0], vprefix_gt[0])
            assert vprefix_gt.shape[-1] == 1

            if self.nets._cfg.predict_q:
                for i in range(self.horizon):
                    vtarg[i] = vtarg[i] * (self.nets._cfg.gamma ** (i+1)) + vprefix_gt[i].mean(axis=-1, keepdim=True)
                vprefix_gt = torch.zeros_like(vprefix_gt) # do not predit vprefix

            assert vprefix_gt.shape[-1] == 1
            vtarg = vtarg.min(axis=-1, keepdims=True)[0] # predict value

        # update the dynamics model ..
        traj = self.nets.inference(obs, None, self.horizon, action) # by default, just rollout for horizon steps ..
        states = traj['states']

        _, vpred, value_prefix = self.nets.predict_values(traj['hidden'], None, None)

        horizon_weights = (self._cfg.rho ** torch.arange(self.horizon, device=obs.device))[:, None]
        horizon_weights = horizon_weights / horizon_weights.sum()
        def hmse(a, b):
            assert a.shape[:-1] == b.shape[:-1], f'{a.shape} vs {b.shape}'
            if a.shape[-1] != b.shape[-1]:
                assert b.shape[-1] == 1 and (a.shape[-1] in [1, 2]), f'{a.shape} vs {b.shape}'
            h = horizon_weights[:len(a)]
            return (((a-b)**2).mean(axis=-1) * h).sum(axis=0)
        
        dyna_loss = {'state': hmse(states, state_gt), 'value': hmse(vpred, vtarg), 'prefix': hmse(value_prefix, vprefix_gt)}
        loss = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss])
        assert loss.shape == weights.shape
        self.dyna_optim.optimize((loss * weights).sum(axis=0)/weights.sum(axis=0))
        logger.logkvs_mean({k: float(v.mean()) for k, v in dyna_loss.items()}, prefix='dyna/')

        # update the value network ..
        value = self.nets.value(obs, None, self.horizon, alpha=0., action_penalty=self._cfg.action_penalty)[0]
        assert value.shape[-1] in [1, 2]
        actor_loss = -value[..., 0].mean(axis=0)
        self.actor_optim.optimize(actor_loss)
        logger.logkv_mean('actor', float(actor_loss))

        if update_target:
            ema(self.nets, self.target_nets, self._cfg.tau)

        with torch.no_grad():
            buffer.update_priorities(idxs, loss[:, None])

    def inference(self, n_step):
        obs, timestep = self.start(self.env, reset=True)
        assert (timestep == 0).all()
        transitions = []

        for _ in range(n_step):
            transition = dict(obs = obs)
            pd = self.nets.policy(obs, None)
            scale = pd.dist.scale
            logger.logkvs_mean({'std_min': float(scale.min()), 'std_max': float(scale.max())})
            a, _ = pd.rsample()
            data, obs = self.step(self.env, a)

            transition.update(**data, a=a)
            transitions.append(transition)

        return Trajectory(transitions, len(obs), n_step)

    def run_rpgm(self, max_epoch=None):
        logger.configure(dir=self._cfg.path)
        env = self.env

        steps = self.env.max_time_steps
        epoch_id = 0
        update_step = 0
        while True:
            if max_epoch is not None and epoch_id >= max_epoch:
                break

            with torch.no_grad():
                traj = self.inference(steps)
                self.buffer.add(traj)

            for _ in tqdm.trange(min(steps, self._cfg.update_step)):
                update_step += 1
                self.update(self.buffer, update_step % self._cfg.update_freq == 0)

            a = traj.get_tensor('a')
            logger.logkv_mean('a_max', float(a.max()))
            logger.logkv_mean('a_min', float(a.min()))

            self.call_hooks(locals())
            print(traj.summarize_epsidoe_info())

            logger.dumpkvs()
