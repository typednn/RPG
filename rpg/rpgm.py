# model-based verison RPG
import tqdm
import copy
import numpy as np
import torch
from tools.config import Configurable
from tools.nn_base import Network
from tools.utils import RunningMeanStd, mlp, orthogonal_init, Seq, logger, totensor, Identity, ema, CatNet
from tools.optim import LossOptimizer
from .common_hooks import RLAlgo, build_hooks
from .buffer import ReplayBuffer
from .traj import Trajectory
from .models import MLPDynamics
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

        done_fn,

        cfg=None,
        gamma=0.99,
        lmbda=0.97,
        predict_q=False,

        lmbda_last=False,


        horizon=1,
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

        self.done_fn = done_fn
            

        v_nets = [enc_s, enc_z, value_fn]
        d_nets = [enc_a, init_h, dynamic_fn, state_dec, value_prefix]
        self.value_nets = torch.nn.ModuleList(v_nets)
        self.dynamics = torch.nn.ModuleList(d_nets)


        weights = []
        lmbda = 1
        sum_lmbda = 0.
        for _ in range(horizon):
            weights.append(lmbda)
            sum_lmbda += lmbda
            lmbda *= self._cfg.lmbda
        weights = torch.tensor(weights, dtype=torch.float32)

        if lmbda_last:
            weights[-1] += (1./(1-self._cfg.lmbda) - sum_lmbda) 
        weights = weights / weights.sum()
        self._weights = torch.nn.Parameter(np.log(weights), requires_grad=True)

    @property
    def weights(self):
        return torch.softmax(self._weights, 0)


    def policy(self, obs, z):
        assert z is None
        obs = totensor(obs, self.device)
        s = self.enc_s(obs)
        return self.pi_a(s, self.enc_z(z))

    def value_from_obs(self, obs, z):
        assert not self._cfg.predict_q
        s = self.enc_s(obs)
        return self.value_fn(s, self.enc_z(z))

    def inference(self, obs, z, step, a_seq=None, pi=None):
        """
        s      a_1    a_2    ...
        |      |      |
        h_0 -> h_1 -> h_2 -> h_3 -> h_4
               |      |      |      |   
               o_1    o_2    o_3    o_4
             / |  
            s1 r1    
            |
            v1
        """
        z_embed = self.enc_z(z)
        assert z_embed is None
        s = self.enc_s(obs)
        h = self.init_h(s)
        h = h.reshape(len(s), self.dynamic_fn.layer, -1).permute(1, 0, 2).contiguous() # GRU of layer 2
        if pi is None: pi = self.pi_a

        hidden, logp_a, actions, states = [], [], [], []
        for idx in range(step):
            assert self.pi_z is None, "this means that we will not update z, anyway.."

            if a_seq is None:
                a, logp = pi(s, z_embed).rsample()
            else:
                a, logp = a_seq[idx], torch.zeros((len(obs),), device=self.device)

            o, h = self.dynamic_fn(self.enc_a(a)[None, :].expand_as(h), h)
            o = o[-1]
            assert o.shape[0] == a.shape[0] 
            s = self.state_dec(o) # predict the next hidden state ..

            hidden.append(o)
            actions.append(a)
            logp_a.append(logp[..., None])
            states.append(s)

        ss = torch.stack
        return dict(hidden=ss(hidden), actions=ss(actions), logp_a=ss(logp_a), states=ss(states), z_embed=None, init_s=s)

    def predict_values(self, ss, hidden, z_embed, rewards=None):
        value_prefix = self.value_prefix(hidden) # predict reward prefix

        if self._cfg.predict_q:
            prefix = torch.zeros_like(value_prefix)
        else:
            prefix = value_prefix

        if rewards is not None and (rewards is not 0):
            prefix = prefix + compute_value_prefix(rewards, self._cfg.gamma)

        if self._cfg.predict_q:
            values = self.value_fn(hidden, z_embed) # predict V(s, z_{t-1})
        else:
            values = self.value_fn(ss, z_embed) # use the invariant of the value function ..

        weights = self.weights
        assert not self._cfg.predict_q
        exepected_values = values
        if self.done_fn is not None:
            dones = torch.sigmoid(self.done_fn(hidden))
            assert dones.shape == prefix.shape
            not_done = 1 - dones
            alive = torch.cumprod(not_done, 0)

            r = prefix.clone()
            r[1:] = r[1:] - r[:-1] # get the gamma decayed rewards ..

            prefix = (r * alive).cumsum(0)
            exepected_values = exepected_values * alive
        else:
            dones = None
            #alive = torch.ones((len(hidden),), device=self.device)[:, None, None]

        gamma = 1
        vpreds = []
        for i in range(len(hidden)):
            if self._cfg.predict_q:
                vpred = exepected_values[i] + prefix[i] # add the additional rewards like entropies 
            else:
                vpred = (prefix[i] + exepected_values[i] * gamma * self._cfg.gamma)

            vpreds.append(vpred)
            gamma *= self._cfg.gamma

        vpreds = torch.stack(vpreds)
        v = (vpreds * weights[:, None, None]).sum(axis=0)
        return dict(value=v, next_values=values, value_prefix=value_prefix, vpreds=vpreds, dones=dones)

    def value(self, obs, z, horizon, alpha=0, action_penalty=0., pi=None):
        traj = self.inference(obs, z, horizon, pi=pi)
        entropy_term = -traj['logp_a']
        entropy = entropy_term * alpha if alpha > 0 else 0
        penalty = action_penalty * (1 - traj['actions']**2).mean(axis=-1, keepdims=True) if action_penalty > 0 else 0
        extra_reward = entropy + penalty
        values =  self.predict_values(traj['states'], traj['hidden'], traj['z_embed'], extra_reward)
        values['entropy_term'] = entropy_term
        values['penalty'] = penalty
        return values


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
        update_target_freq=2,
        tau=0.005,
        rho=0.97, # horizon decay
        max_update_step=200,
        weights=dict(state=1000., prefix=0.5, value=0.5, done=0.5),
        qnet=GeneralizedQ.dc,
        action_penalty=0.,

        entropy_coef=0.,
        entropy_target=None,
        # critic_weight=0.,
        norm_s_enc=False,
        update_train_step=1,


        # reward_norm=False,  # normalize the reward
        adv_norm = False,  # if adv_norm is True, normalize the values..
        rew_norm = False,


        selfsupervised=False,

        use_target_net=True,

        have_done=False,
    ):
        Configurable.__init__(self)
        RLAlgo.__init__(self, (RunningMeanStd(clip_max=10.) if obs_norm else None), build_hooks(hooks))
        self.rew_norm = RunningMeanStd(clip_max=10.) if rew_norm else None

        obs_space = env.observation_space
        action_dim = env.action_space.shape[0]

        self.horizon = horizon
        self.env = env

        # buffer samples horizon + 1
        self.buffer = ReplayBuffer(obs_space.shape, action_dim, env.max_time_steps, horizon, cfg=buffer)
        nets = self.nets = self.make_network(obs_space, env.action_space).cuda()

        with torch.no_grad():
            self.target_nets = copy.deepcopy(nets).cuda()

        # only optimize pi_a here
        self.actor_optim = LossOptimizer(nets.pi_a, cfg=actor_optim)
        self.dyna_optim = LossOptimizer(torch.nn.ModuleList([nets.dynamics, nets.value_nets]), cfg=actor_optim)

        self.log_alpha = torch.nn.Parameter(
            torch.zeros(1, requires_grad=(entropy_target is not None), device='cuda:0'))
        if entropy_target is not None:
            if entropy_target > 0:
                entropy_target =  - action_dim
            self.entropy_optim = LossOptimizer(self.log_alpha, cfg=actor_optim) #TODO: change the optim ..

        self.update_step = 0

    def evaluate(self, env, steps):
        with torch.no_grad():
            self.start(self.env, reset=True)
            out = self.inference(steps, mode='not_sample')
            self.start(self.env, reset=True)
            return out

    def make_network(self, obs_space, action_space):
        hidden_dim = 256
        latent_dim = 100 # encoding of state ..
        enc_s = mlp(obs_space.shape[0], hidden_dim, latent_dim) # TODO: layer norm?
        if self._cfg.norm_s_enc:
            enc_s = Seq(enc_s, torch.nn.LayerNorm(latent_dim))
        enc_z = Identity()
        enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)

        layer = 1
        init_h = mlp(latent_dim, hidden_dim, hidden_dim * layer)
        dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
        dynamics.layer = layer
        # dynamics = MLPDynamics(hidden_dim)
        v_in = hidden_dim if self._cfg.qnet.predict_q else latent_dim
        value = CatNet(
            Seq(mlp(v_in, hidden_dim, 1)),
            Seq(mlp(v_in, hidden_dim, 1)),
        ) #layer norm, if necesssary
        done_fn = mlp(hidden_dim, hidden_dim, 1) if self._cfg.have_done else None

        value_prefix = Seq(mlp(hidden_dim, hidden_dim, 1))
        state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..

        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = Seq(mlp(latent_dim, hidden_dim, head.get_input_dim()), head)
        network = GeneralizedQ(enc_s, enc_a, enc_z, pi_a, None, init_h, dynamics, state_dec, value_prefix, value, done_fn, cfg=self._cfg.qnet, horizon=self._cfg.horizon)
        network.apply(orthogonal_init)
        return network
        
    def update(self):
        with torch.no_grad():
            alpha = self._cfg.entropy_coef * torch.exp(self.log_alpha).detach()

        supervised_horizon = self.horizon # not +1 yet

        obs, next_obs, action, reward, done_gt, truncated = self.buffer.sample(self._cfg.batch_size)
        batch_size = obs.shape[0]

        with torch.no_grad():
            pi = None if self._cfg.use_target_net else self.nets.pi_a
            vtarg = self.target_nets.value(next_obs.reshape(-1, *obs.shape[1:]), None,
                self.horizon, alpha=alpha, pi=pi)['value'].reshape(next_obs.shape[0], batch_size, -1)[:supervised_horizon]
            state_gt = self.nets.enc_s(next_obs)[:supervised_horizon] # use the current model ..
            vprefix_gt = compute_value_prefix(reward, self.nets._cfg.gamma)[:supervised_horizon]
            assert torch.allclose(reward[0], vprefix_gt[0])


            if self.rew_norm is not None:
                # vtarg = vtarg * self.rew_norm.std # denormalize it.
                self.rew_norm.update(vprefix_gt) # 'normalize reward during update ..'
                vprefix_gt = vprefix_gt / self.rew_norm.std

            assert vprefix_gt.shape[-1] == 1

            if self.nets._cfg.predict_q:
                for i in range(self.horizon):
                    vtarg[i] = vtarg[i] * (self.nets._cfg.gamma ** (i+1)) + vprefix_gt[i].mean(axis=-1, keepdim=True)
                vprefix_gt = torch.zeros_like(vprefix_gt) # do not predit vprefix

            assert vprefix_gt.shape[-1] == 1
            vtarg = vtarg.min(axis=-1, keepdims=True)[0] # predict value
        
        if self._cfg.selfsupervised:
            state_gt = self.nets.enc_s(next_obs)[:supervised_horizon] # pass grad

        # update the dynamics model ..
        traj = self.nets.inference(obs, None, self.horizon, action) # by default, just rollout for horizon steps ..
        states = traj['states']
        out = self.nets.predict_values(states, traj['hidden'], None, None)
        vpred, value_prefix = out['next_values'], out['value_prefix']

        horizon_weights = torch.ones_like(self.nets.weights.detach())[:, None]
        horizon_weights = horizon_weights / horizon_weights.sum()
        truncated_mask = torch.ones_like(truncated) # we weill not predict state after done ..
        truncated_mask[1:] = 1 - (truncated.cumsum(0)[:-1] > 0).float()
        assert truncated_mask.shape[-1] == 1

        def hmse(a, b):
            assert a.shape[:-1] == b.shape[:-1], f'{a.shape} vs {b.shape}'
            if a.shape[-1] != b.shape[-1]:
                assert b.shape[-1] == 1 and (a.shape[-1] in [1, 2]), f'{a.shape} vs {b.shape}'
            h = horizon_weights[:len(a)] * truncated_mask[..., 0]
            difference = ((a-b)**2).mean(axis=-1)
            assert difference.shape == h.shape
            return (difference * h).sum(axis=0)
        
        dyna_loss = {'state': hmse(states, state_gt), 'value': hmse(vpred, vtarg), 'prefix': hmse(value_prefix, vprefix_gt)}

        if self._cfg.have_done:
            loss = torch.nn.functional.binary_cross_entropy(
                out['dones'], done_gt, reduction='none'
            ).mean(axis=-1) * truncated_mask[..., 0]
            dyna_loss['done'] = (loss * horizon_weights[:len(done_gt)]).sum(axis=0)
            logger.logkv_mean('done_acc', ((out['dones'] > 0.5) == done_gt).float().mean())

        loss = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss])
        dyna_loss_total = loss.mean(axis=0)

        # # value
        # if self._cfg.critic_weight > 0.:
        #     with torch.no_grad():
        #         value = self.target_nets.value(obs, None, self.horizon, alpha=alpha)['value'].min(axis=-1, keepdims=True)[0]
        #         vtarg = torch.cat((value[None,:], vtarg), axis=0) # add the value of the first state ..
        #     vpred = self.nets.value_from_obs(torch.cat((obs[None,:], next_obs[:supervised_horizon]), axis=0), z=None)
        #     critic_loss = ((vpred - vtarg) ** 2).mean(axis=-1).mean(axis=0) # do not weight it again ..
        #     critic_loss = (critic_loss * weights).sum(axis=0)/weights.sum(axis=0)
        #     #self.value_optim.optimize(critic_loss)
        #     logger.logkv_mean('critic', float(critic_loss))
        #     dyna_loss_total += self._cfg.critic_weight * critic_loss # additional value predict .. 

        self.dyna_optim.optimize(dyna_loss_total)
        logger.logkvs_mean({k: float(v.mean()) for k, v in dyna_loss.items()}, prefix='dyna/')
        logger.logkv_mean('dyna_loss', float(dyna_loss_total))


        # update the actor network ..
        samples = self.nets.value(obs, None, self.horizon, alpha=alpha, action_penalty=self._cfg.action_penalty)
        value = samples['value']

        assert value.shape[-1] in [1, 2]
        actor_loss = -value[..., 0]
        if self._cfg.adv_norm:
            actor_loss = actor_loss / (actor_loss.std(axis=0).detach() + 1e-8)
        actor_loss = actor_loss.mean(axis=0)
        self.actor_optim.optimize(actor_loss)
        logger.logkv_mean('actor', float(actor_loss))

        if self._cfg.entropy_target is not None:
            entropy_loss = -torch.mean(self.log_alpha.exp() * (self._cfg.entropy_target - samples['entropy_term'].detach()))
            self.entropy_optim.optimize(entropy_loss)
        logger.logkv_mean('alpha', float(alpha))
        logger.logkv_mean('entropy_term', float(samples['entropy_term'].mean()))


        self.update_step += 1
        # previous the update_freq 
        if self.update_step % self._cfg.update_target_freq == 0:
            ema(self.nets, self.target_nets, self._cfg.tau)


    def inference(self, n_step, mode='sample'):
        with torch.no_grad():
            obs, timestep = self.start(self.env)
            transitions = []

        r = tqdm.trange if self._cfg.update_train_step > 0 else range 
        for idx in r(n_step):
            with torch.no_grad():
                transition = dict(obs = obs)
                pd = self.nets.policy(obs, None)
                scale = pd.dist.scale
                logger.logkvs_mean({'std_min': float(scale.min()), 'std_max': float(scale.max())})
                if mode == 'sample':
                    a, _ = pd.rsample()
                else:
                    a = pd.dist.loc.detach().cpu().numpy()
                data, obs = self.step(self.env, a)

                transition.update(**data, a=a)
                transitions.append(transition)

            if self.buffer.total_size() > 10000 and self._cfg.update_train_step > 0:
                if idx % self._cfg.update_train_step == 0:
                    self.update()

        return Trajectory(transitions, len(obs), n_step)

    def run_rpgm(self, max_epoch=None):
        logger.configure(dir=self._cfg.path, format_strs=["stdout", "log", "csv", 'tensorboard'])
        env = self.env

        steps = self.env.max_time_steps
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
