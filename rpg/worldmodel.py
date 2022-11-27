"""
# model-based verison RPG
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
import torch
from tools.utils import totensor
from tools.nn_base import Network
from .utils import lmbda_decay_weight, masked_temporal_mse
from tools.config import Configurable
from tools.optim import LossOptimizer
from tools.utils import logger
from torch.nn.functional import binary_cross_entropy as bce


class GeneralizedQ(torch.nn.Module):
    def __init__(
        self,
        enc_s, enc_a,
        init_h,
        dynamic_fn,
        state_dec,
        reward_predictor,
        q_fn,
        done_fn,

        gamma,

        # gamma=0.99,
        # lmbda=0.97,
        # lmbda_last=False,
        # horizon=1,
        # detach_hidden=True,  # don't let the done to affect the hidden learning ..
    ) -> None:
        super().__init__()
        # self.pi_a, self.pi_z = pi_a, pi_z
        self.enc_s, self.enc_a = enc_s, enc_a

        self.init_h = init_h
        self.dynamic_fn: torch.nn.GRU = dynamic_fn

        self.state_dec = state_dec
        self.reward_predictor = reward_predictor
        self.q_fn = q_fn

        self.done_fn = done_fn
        self.gamma = gamma

    @property
    def weights(self):
        return torch.softmax(self._weights, 0)

    def policy(self, obs, prevz, timestep, pi_z, pi_a):
        obs = totensor(obs, self.device)
        prevz = totensor(prevz, self.device, dtype=None)
        timestep = totensor(timestep, self.device, dtype=None)
        s = self.enc_s(obs, timestep=timestep)
        z = pi_z(s, prevz, timestep).z
        a = pi_a(s, z).a
        return a, z.detach().cpu().numpy()

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
                z, _logp_z, z_new, logp_z_new, _entz = pi_z(s, z, timestep=timestep)
                logp_z.append(_logp_z[..., None])
                entz.append(_entz[..., None])
                z_seq.append(z)
            z = z_seq[idx]

            if len(a_seq) <= idx:
                a, _logp_a = pi_a(s, z)
                logp_a.append(_logp_a[..., None])
                a_seq.append(a)

            a_embed = self.enc_a(a_seq[idx])
            a_embeds.append(a_embed)
            o, h = self.dynamic_fn(a_embed[None, :], h)
            assert torch.allclose(o[-1], h)
            s = self.state_dec(h[-1]) # predict the next hidden state ..

            # hidden.append(h[-1])
            states.append(s)

            timestep = timestep + 1

        states = torch.stack(states)
        a_embeds = torch.stack(a_embeds)
        out = dict(state=states, reward=self.reward_predictor(states[1:], a_embeds))

        if sample_a:
            a_seq = out['a'] = torch.stack(a_seq)
            out['logp_a'] = torch.stack(logp_a)

        if sample_z:
            z_seq = out['z'] = torch.stack(z_seq)
            out['logp_z'] = torch.stack(logp_z)
            out['ent_z'] = torch.stack(entz)
            
        dones = None
        if self.done_fn is not None:
            #done_inp = torch.concat((states[1:], a_embeds), -1)
            done_inp = states[1:]
            detach_hidden = True
            if detach_hidden:
                done_inp = done_inp.detach()
            out['done'] = dones = torch.sigmoid(self.done_fn(done_inp))

        q_values, values = self.q_fn(states[:-1], z_seq, a_seq, new_s=states[1:], r=out['reward'], done=dones, gamma=self.gamma)

        if dones is not None:
            from tools.utils import logger 
            logger.logkv_mean('mean_pred_done', dones.mean().item())

        if sample_z:
            rewards, extra_rewards, infos = intrinsic_reward(out) # return rewards, (ent_a,ent_z)
            #assert extra_rewards.shape[-1] == 3, "enta, entz, info"
            out.update(infos)

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
            #out['value'] = (vpreds * self.weights[:, None, None]).sum(axis=0)
            out['vpreds'] = vpreds
            out['extra_rewards'] = extra_rewards
            assert out['value'].shape[-1] == 2, "must be double q learning .."

        out['q_value'] = q_values
        out['pred_values'] = values

        return out




class HiddenDynamicNet(Network, GeneralizedQ):
    def __init__(self, 
        obs_space, action_space, z_space,
        cfg=None, 
        qmode='Q',
        gamma=0.99,
        lmbda=0.97,
        lmbda_last=False,
        horizon=1,
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

        if state_layer_norm:
            args.append(torch.nn.LayerNorm(latent_dim, elementwise_affine=False))
        if state_batch_norm:
            args.append(BN(latent_dim))
        # assert len(args) == 0

        if no_state_encoder:
            enc_s = TimedSeq(Identity(), *args) # encode state with time step ..
            latent_dim = obs_space.shape[0]
        else:
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
        self.enc_z = ZTransform(z_space)
        from .soft_actor_critic import SoftQPolicy, ValuePolicy


        action_dim = action_space.shape[0]
        if qmode == 'Q':
            q_fn = SoftQPolicy(state_dim, action_dim, z_space, self.enc_z, hidden_dim)
        else:
            q_fn = ValuePolicy(state_dim, action_dim, z_space, self.enc_z, hidden_dim, zero_done_value=self._cfg.zero_done_value)


        weights = lmbda_decay_weight(lmbda, horizon, lmbda_last=lmbda_last)
        self.weights = torch.softmax(torch.log(weights), -1)
        
        Network.__init__(self, cfg)
        GeneralizedQ.__init__(self, enc_s, enc_a, init_h, dynamics, state_dec, reward_predictor, q_fn, done_fn)

    def inference(self, obs, z, timestep, step, pi_z=None, pi_a=None, z_seq=None, a_seq=None, intrinsic_reward=None):
        out = super().inference(obs, z, timestep, step, pi_z, pi_a, z_seq, a_seq, intrinsic_reward)
        if 'vpreds' in out:
            assert self.weights.shape[0] == len(out['vpreds'])
            out['values'] = self.weights[:, None, None] * out['vpreds']
        return out


class DynamicsLearner(LossOptimizer):
    # optimizer of the dynamics model ..
    def __init__(
        self, models: HiddenDynamicNet,
        pi_a, pi_z, intrinsic_reward, target_net,
        
        cfg=None,
        target_horizon=None, zero_done_value=False, have_done=False,
        weights=dict(state=1000., reward=0.5, q_value=0.5, done=1.),

    ):
        super().__init__(models, cfg)
        self.nets = models
        self.horizon = models._cfg.horizon

        self.pi_a = pi_a
        self.pi_z = pi_z
        self.intrinsic_reward = intrinsic_reward
        self.target_net = target_net


    def learn_dynamics(self, obs_seq, timesteps, action, reward, done_gt, truncated_mask, prev_z):
        pred_traj = self.nets.inference(obs_seq[0], prev_z[0], timesteps[0], self.horizon, a_seq=action, z_seq=prev_z[1:]) 

        with torch.no_grad():

            gt = dict(reward=reward)
            dyna_loss = dict()

            if isinstance(obs_seq, torch.Tensor):
                batch_size = len(obs_seq[0])
                next_obs = obs_seq[1:].reshape(-1, *obs_seq.shape[2:])
            else:
                assert isinstance(obs_seq[0], dict)
                next_obs = {}
                for k in obs_seq[0]:
                    # [T, B, ...]
                    next_obs[k] = torch.stack([v[k] for v in obs_seq[1:]])
                batch_size = next_obs[k].shape[1]

                for k, v in next_obs.items():
                     # print(k, v.shape)
                     next_obs[k] = v.reshape(-1, *v.shape[2:])
                # exit(0)

            z_seq = prev_z[1:].reshape(-1, *prev_z.shape[2:])
            # zz = z_seq.reshape(-1).detach().cpu().numpy().tolist(); print([zz.count(i) for i in range(6)])
            next_timesteps = timesteps[1:].reshape(-1)

            samples = self.target_net.inference(
                next_obs, z_seq, next_timesteps, self._cfg.target_horizon or self.horizon,
                pi_z = self.pi_z, pi_a = self.pi_a, intrinsic_reward = self.intrinsic_reward,
            )
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
        self.optimize(dyna_loss_total)
        info = {'dyna_' + k + '_loss': float(v.mean()) for k, v in dyna_loss.items()}
        logger.logkv_mean('dyna_total_loss', float(dyna_loss_total.mean()))
        logger.logkvs_mean(info)

        s = output['state']
        logger.logkv_mean('state_mean', float(s.mean()))
        logger.logkv_mean('state_min', float(s.min()))
        logger.logkv_mean('state_max', float(s.max()))