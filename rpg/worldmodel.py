import torch
from tools.nn_base import Network
from .utils import masked_temporal_mse
from tools.optim import LossOptimizer
from tools.utils import logger
from torch.nn.functional import binary_cross_entropy as bce

from typing import  Union
from .critic import SoftQPolicy, ValuePolicy


class GeneralizedQ(torch.nn.Module):
    def __init__(
        self, enc_s, enc_a, init_h, dynamic_fn, state_dec, reward_predictor, q_fn, done_fn, gamma,
    ) -> None:
        super().__init__()
        # self.pi_a, self.pi_z = pi_a, pi_z
        self.enc_s = enc_s
        self.enc_a = enc_a
        self.init_h = init_h
        self.dynamic_fn: torch.nn.GRU = dynamic_fn
        self.state_dec = state_dec
        self.reward_predictor = reward_predictor
        self.q_fn: Union[SoftQPolicy, ValuePolicy] = q_fn
        self.done_fn = done_fn
        self.gamma = gamma

    def core(self, h, a):
        a_embed = self.enc_a(a)
        o, h = self.dynamic_fn(a_embed[None, :], h)
        assert torch.allclose(o[-1], h)
        s = self.state_dec(h[-1]) # predict the next hidden state ..
        return h, s, a_embed

    def pred_done(self, traj):
        dones = None
        if self.done_fn is not None:
            states = traj['state']
            #done_inp = torch.concat((states[1:], a_embeds), -1)
            done_inp = states[1:]
            detach_hidden = True
            if detach_hidden:
                done_inp = done_inp.detach()
            dones = torch.sigmoid(self.done_fn(done_inp))
            from tools.utils import logger 
            logger.logkv_mean('mean_pred_done', dones.mean().item())

        traj['done'] = dones
        return dones

    def pred_rewards(self, traj):
        traj['reward'] = self.reward_predictor(traj['state'][1:], traj['a_embed'])
        return traj['reward']

    def pred_q_value(self, traj):
        states = traj['state']
        z_seq = traj['z']
        a_seq = traj['a']
        reward = traj['reward']
        dones = traj['done']
        q_values, values = self.q_fn(states[:-1], z_seq, a_seq, new_s=states[1:], r=reward, done=dones, gamma=self.gamma)

        traj['q_value'] = q_values
        traj['pred_values'] = values
        return q_values

    def inference_with_actions(self, obs, timestep, step, z_seq, a_seq):
        s = self.enc_s(obs, timestep=timestep)
        states = [s]
        # for dynamics part ..
        h = self.init_h(s)[None,:]
        a_embeds = []
        for idx in range(step):
            h, s, a_embed = self.core(h, a_seq[idx])
            a_embeds.append(a_embed)
            states.append(s)
        out = dict(state=torch.stack(states), a_embed=torch.stack(a_embeds), z=z_seq)
        self.pred_done(out)
        self.pred_rewards(out)
        self.pred_q_value(out)
        return out

    def inference(
        self, obs, z, timestep, step,
        pi_z=None, pi_a=None, z_seq=None, a_seq=None, 
        intrinsic_reward=None,
    ):
        # z_seq is obs -> z -> a
        assert a_seq is not None or pi_a is not None
        assert z_seq is not None or pi_z is not None

        if isinstance(obs, torch.Tensor):
            assert timestep.shape == (len(obs),)

        sample_z = (z_seq is None)
        if sample_z:
            z_seq, logp_z, entz = [], [], []

        sample_a = (a_seq is None)
        if sample_a:
            a_seq, logp_a = [], []

        s = self.enc_s(obs, timestep=timestep)
        states = [s]

        # for dynamics part ..
        h = self.init_h(s)[None,:]
        a_embeds = []


        for idx in range(step):
            if len(z_seq) <= idx:
                z, _logp_z, z_new, logp_z_new, _entz = pi_z(s, z, prev_action=z, timestep=timestep)
                logp_z.append(_logp_z[..., None])
                entz.append(_entz[..., None])
                z_seq.append(z)
            z = z_seq[idx]

            if len(a_seq) <= idx:
                a, _logp_a, _enta = pi_a(s, z)
                logp_a.append(_logp_a[..., None])
                a_seq.append(a)
            
            h, s, a_embed = self.core(h, a_seq[idx])
            a_embeds.append(a_embed)
            states.append(s)

            timestep = timestep + 1

        out = dict(state=torch.stack(states), a_embed=torch.stack(a_embeds))
        self.pred_rewards(out)

        if sample_a:
            a_seq = out['a'] = torch.stack(a_seq)
            out['logp_a'] = torch.stack(logp_a)
            out['ent_a'] = -out['logp_a']
        else:
            out['a'] = a_seq

        if sample_z:
            z_seq = out['z'] = torch.stack(z_seq)
            out['logp_z'] = torch.stack(logp_z)
            out['ent_z'] = torch.stack(entz)
        else:
            out['z'] = z_seq
            
        dones = self.pred_done(out)
        #q_values, values = self.q_fn(states[:-1], z_seq, a_seq, new_s=states[1:], r=reward, done=dones, gamma=self.gamma)
        q_values = self.pred_q_value(out)

        if sample_z:
            rewards, extra_rewards = intrinsic_reward(out) # return rewards, (ent_a,ent_z)
            vpreds = []
            discount = 1
            prefix = 0.
            for i in range(len(rewards)):
                prefix = prefix + extra_rewards[i].sum(axis=-1, keepdims=True) * discount
                vpreds.append(prefix + q_values[i] * discount)
                prefix = prefix + rewards[i] * discount
                discount = discount * self.gamma
                if dones is not None:
                    assert dones.shape[-1] == 1
                    discount = discount * (1 - dones[i]) # the probablity of not done ..
            vpreds = torch.stack(vpreds)
            out['vpreds'] = vpreds

        return out


class HiddenDynamicNet(Network, GeneralizedQ):
    def __init__(self, 
        obs_space, action_space, z_space,
        cfg=None, 
        qmode='Q',
        gamma=0.99,
        # lmbda_last=False,
        detach_hidden=True,  # don't let the done to affect the hidden learning ..


        state_layer_norm=False,
        state_batch_norm=False,
        no_state_encoder=False,

        dynamic_type='normal',
        state_dim=100,

        have_done=False,
        hidden_dim=256,
    ):
        # Encoder
        # TODO: layer norm?
        from .utils import ZTransform
        import gym
        from tools.utils import TimedSeq, Identity, mlp, Seq

        args = []
        class BN(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.bn = torch.nn.BatchNorm1d(dim)

            def forward(self, x):
                if len(x.shape) == 3:
                    return self.bn(x.transpose(1, 2)).transpose(1, 2)
                return self.bn(x)


        if no_state_encoder:
            enc_s = TimedSeq(Identity(), *args) # encode state with time step ..
            latent_dim = obs_space.shape[0]
        else:
            latent_dim = state_dim
            if state_layer_norm:
                args.append(torch.nn.LayerNorm(latent_dim, elementwise_affine=False))
            if state_batch_norm:
                args.append(BN(latent_dim))

            if not isinstance(obs_space, dict):
                enc_s = TimedSeq(mlp(obs_space.shape[0], hidden_dim, latent_dim), *args) # encode state with time step ..
            else:
                from nn.modules.point import PointNet
                assert not state_layer_norm and not state_batch_norm
                enc_s = PointNet(obs_space, output_dim=latent_dim)

        self.state_dim = latent_dim
        # enc_z = ZTransform(z_space)

        # dynamics
        layer = 1 # gru layers ..
        if dynamic_type == 'normal':
            if isinstance(action_space, gym.spaces.Box):
                enc_a = mlp(action_space.shape[0], hidden_dim, hidden_dim)
            else:
                enc_a = torch.nn.Embedding(action_space.n, hidden_dim)
            a_dim = hidden_dim

            init_h = mlp(latent_dim, hidden_dim, hidden_dim)
            dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
            state_dec = mlp(hidden_dim, hidden_dim, latent_dim) # reconstruct obs ..
        elif dynamic_type == 'tiny':
            raise NotImplementedError
            if isinstance(action_space, gym.spaces.Box):
                enc_a = torch.nn.Linear(action_space.shape[0], hidden_dim)
                a_dim = hidden_dim
            else:
                enc_a = torch.nn.Embedding(action_space.n, hidden_dim)
                a_dim = hidden_dim
            init_h = torch.nn.Linear(latent_dim, hidden_dim)
            dynamics = torch.nn.GRU(hidden_dim, hidden_dim, layer)
            state_dec = torch.nn.Linear(hidden_dim, latent_dim)
        else:
            raise NotImplementedError

        # reawrd and done
        reward_predictor = Seq(mlp(a_dim + latent_dim, hidden_dim, 1)) # s, a predict reward .. 
        done_fn = mlp(latent_dim, hidden_dim, 1) if have_done else None

        # Q
        enc_z = ZTransform(z_space)
        action_dim = action_space.shape[0]
        q_fn = (SoftQPolicy if qmode == 'Q' else ValuePolicy)(state_dim, action_dim, z_space, enc_z, hidden_dim)

        Network.__init__(self, cfg)
        GeneralizedQ.__init__(self, enc_s, enc_a, init_h, dynamics, state_dec, reward_predictor, q_fn, done_fn, gamma)
        self.enc_z = enc_z

    def inference(self, *args, **kwargs):
        out = super().inference(*args, **kwargs)
        if 'vpreds' in out:
            #out['value'] = out['vpreds'].mean(axis=0)
            lmbda = 1.
            s = 0.
            total = 0
            #for i in range(len(out['vpreds'])):
            for v in out['vpreds']:
                total = total + lmbda * v
                s += lmbda
                lmbda = lmbda * 0.97
            out['value'] = total / s
            assert out['value'].shape[-1] == 2, "must be double q learning .."
        return out


class DynamicsLearner(LossOptimizer):
    # optimizer of the dynamics model ..
    def __init__(
        self, models: HiddenDynamicNet,
        pi_a, pi_z, intrinsic_reward,
        
        cfg=None,
        target_horizon=None, have_done=False,
        weights=dict(state=1000., reward=0.5, q_value=0.5, done=1.),

        max_grad_norm=1.,
        lr=3e-4,
    ):
        super().__init__(models, cfg)
        self.pi_a = pi_a
        self.pi_z = pi_z
        self.intrinsic_reward = intrinsic_reward

        self.nets = models
        with torch.no_grad():
            import copy
            self.target_net = copy.deepcopy(self.nets)


    def get_flatten_next_obs(self, obs_seq):
        if isinstance(obs_seq, torch.Tensor):
            next_obs = obs_seq[1:].reshape(-1, *obs_seq.shape[2:])
        else:
            assert isinstance(obs_seq[0], dict)
            next_obs = {}
            for k in obs_seq[0]:
                # [T, B, ...]
                next_obs[k] = torch.stack([v[k] for v in obs_seq[1:]])
            for k, v in next_obs.items():
                    next_obs[k] = v.reshape(-1, *v.shape[2:])
            raise NotImplementedError
        return next_obs

    def learn_dynamics(self, obs_seq, timesteps, action, reward, done_gt, truncated_mask, prev_z):
        horizon = len(obs_seq) - 1
        qnet = self.nets.q_fn

        with torch.no_grad():
            dyna_loss = dict()

            batch_size = action.shape[1]
            next_obs = self.get_flatten_next_obs(obs_seq)
            z_seq = prev_z[1:].reshape(-1, *prev_z.shape[2:])
            next_timesteps = timesteps[1:].reshape(-1)

            self.pi_a.set_mode('target'); self.pi_z.set_mode('target')
            samples = self.target_net.inference(
                next_obs, z_seq, next_timesteps, self._cfg.target_horizon or horizon,
                pi_z=self.pi_z, pi_a=self.pi_a, intrinsic_reward = self.intrinsic_reward,
            )
            self.pi_a.set_mode('train'); self.pi_z.set_mode('train')

            vtarg = samples['value'].min(axis=-1)[0].reshape(-1, batch_size, 1)
            assert reward.shape == vtarg.shape == done_gt.shape, (reward.shape, vtarg.shape, done_gt.shape)

            gt = dict(
                reward=reward,
                q_value=qnet.compute_target(vtarg, reward, done_gt.float(), self.nets.gamma),
                state=samples['state'][0].reshape(-1, batch_size, samples['state'].shape[-1])
            )
            logger.logkv_mean('q_value', float(gt['q_value'].mean()))
            logger.logkv_mean('reward_step_mean', float(reward.mean()))

        pred_traj = self.nets.inference(
            obs_seq[0], prev_z[0], timesteps[0], len(action), a_seq=action, z_seq=prev_z[1:]) 

        output = dict(state=pred_traj['state'][1:], q_value=qnet.get_predict(pred_traj),  reward=pred_traj['reward'])
        for k in ['state', 'q_value', 'reward']:
            dyna_loss[k] = masked_temporal_mse(output[k], gt[k], truncated_mask) / horizon

        # assert truncated_mask.all()
        if self._cfg.have_done:
            assert done_gt.shape == pred_traj['done'].shape
            loss = bce(pred_traj['done'], done_gt, reduction='none').mean(axis=-1) # predict done all the way ..
            dyna_loss['done'] = (loss * truncated_mask).sum(axis=0)
            logger.logkv_mean('done_acc', ((pred_traj['done'] > 0.5) == done_gt).float().mean())

        dyna_loss_total = sum([dyna_loss[k] * self._cfg.weights[k] for k in dyna_loss]).mean(axis=0)
        self.optimize(dyna_loss_total)

        info = {'dyna_' + k + '_loss': float(v.mean()) for k, v in dyna_loss.items()}
        logger.logkv_mean('dyna_total_loss', float(dyna_loss_total.mean()))
        logger.logkvs_mean(info)

        s = output['state']
        logger.logkv_mean('state_mean', float(s.mean()))
        logger.logkv_mean('state_min', float(s.min()))
        logger.logkv_mean('state_max', float(s.max()))

    def ema(self, tau):
        from tools.utils import ema
        ema(self.nets, self.target_net, tau)