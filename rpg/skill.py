# code for learning skill.
# Let's consider the simplest skill learning framework
# we assume a strong on-policy properties, or we assume that we have a very large on-policy buffer. 
# don't know how to do it right now .. have to think about it carefully.. there is not way to really encode a long horizon trajectory
# so ideally we have to do it in a way that we can encode a long horizon trajectory.

import torch
from .soft_rpg import *
from .soft_actor_critic import GaussianPolicy


class SkillLearning(Trainer):
    # gaussian latent ..
    def __init__(self,
                 env: Union[GymVecEnv, TorchEnv],
                 cfg=None, z_dim=0, z_cont_dim=0, buffer=dict(store_z=True)
            ):
        super().__init__(env)

        self.pi_a_optim = LossOptimizer(self.nets.pi_a, cfg=self._cfg.optim)
        self.pi_z_optim = LossOptimizer(self.nets.pi_z, cfg=self._cfg.optim)


    def make_network(self, obs_space, action_space, z_space):
        hidden_dim = 256
        state_dim = 100
        action_dim = action_space.shape[0]

        enc_s, enc_z, enc_a, init_h, dynamics, reward_predictor, done_fn, state_dec = self.make_dynamic_network(
            obs_space, action_space, z_space, hidden_dim, state_dim)
        state_dim = self.state_dim

        q_fn = ValuePolicy(state_dim, action_dim, z_space, enc_z, hidden_dim, zero_done_value=self._cfg.zero_done_value)

        head = DistHead.build(action_space, cfg=self._cfg.head)
        pi_a = PolicyA(state_dim, hidden_dim, enc_z, head)

        if self._cfg.z_dim == 0:
            pi_z = GaussianPolicy(state_dim, hidden_dim, self.z_space, cfg=self._cfg.pi_z)
        else:
            pi_z = SoftPolicyZ(state_dim, hidden_dim, enc_z, cfg=self._cfg.pi_z)

        network = GeneralizedQ(
            enc_s, enc_a, pi_a, pi_z,
            init_h, dynamics, state_dec, reward_predictor, q_fn,
            done_fn, None, cfg=self._cfg.worldmodel, gamma=self._cfg.gamma, lmbda=self._cfg.lmbda, horizon=self._cfg.horizon
        )
        network.apply(orthogonal_init)
        info_net = self.make_intrinsic_reward(
            obs_space, action_space, z_space, hidden_dim, state_dim
        )
        return network.cuda(), info_net

    def update_pi_a(self):
        if self.update_step % self._cfg.actor_delay == 0:
            obs_seq, timesteps, action, reward, done_gt, truncated_mask, z = self.buffer.sample(self._cfg.batch_size, horizon=1)
            rollout = self.nets.inference(obs_seq[0], z, timesteps[0], self.horizon)

            loss_a = self.nets.pi_a.loss(rollout)
            logger.logkv_mean('a_loss', float(loss_a))
            self.pi_a_optim.optimize(loss_a)

            enta, _ = self.intrinsic_reward.get_ent_from_traj(rollout)
            self.enta.update(enta)
            logger.logkv_mean('a_alpha', float(self.enta.alpha))
            logger.logkv_mean('a_ent', float(enta.mean()))

            self.intrinsic_reward.update(rollout)

    def update_pi_z(self):
        if self.update_step % self.z_delay == 0:
            o, z, t = self.buffer.sample_start(self._cfg.batch_size)
            assert (t < 1).all()
            rollout = self.nets.inference(o, z, t, self.horizon)

            loss_z = self.nets.pi_z.loss(rollout)
            logger.logkv_mean('z_loss', float(loss_z))
            self.pi_z_optim.optimize(loss_z)

            _, entz = self.intrinsic_reward.get_ent_from_traj(rollout)
            self.entz.update(entz)
            logger.logkv_mean('z_alpha', float(self.entz.alpha))
            logger.logkv_mean('z_ent', float(entz.mean()))


    def update(self):
        self.nets.train()
        self.sync_alpha()

        # learning the low-level will be similar to the others ..
        obs_seq, timesteps, action, reward, done_gt, truncated_mask, z = self.buffer.sample(self._cfg.batch_size)
        # ---------------------- update dynamics ----------------------
        assert len(obs_seq) == len(timesteps) == len(action) + 1 == len(reward) + 1 == len(done_gt) + 1 == len(truncated_mask) + 1
        batch_size = len(obs_seq[0])
        prev_z = z[None, :].expand(len(obs_seq), *z.shape)
        self.learn_dynamics(obs_seq, timesteps, action, reward, done_gt, truncated_mask, prev_z)

        self.update_pi_a()
        self.update_pi_z()

        # first update the actor 

        # # ---------------------- update actor ----------------------
        # rollout = self.nets.inference(obs_seq[0], prev_z[0], timesteps[0], self.horizon)

        # if self.update_step % self.z_delay == 0:
        #     loss_z = self.nets.pi_z.loss(rollout)
        #     if self.update_step % self._cfg.actor_delay == 0:
        #         loss_a = self.nets.pi_a.loss(rollout)
        #         logger.logkv_mean('a_loss', float(loss_a))
        #     else:
        #         loss_a = 0.
        #     logger.logkv_mean('z_loss', float(loss_z))
        #     self.actor_optim.optimize(loss_a + loss_z)

        # enta, entz = self.intrinsic_reward.get_ent_from_traj(rollout)
        # logger.logkv_mean('a_ent', enta.mean())
        # logger.logkv_mean('z_ent', entz.mean())
        # if self.update_step % self.z_delay == 0:
        #     self.entz.update(entz)
        #     logger.logkv_mean('z_alpha', float(self.entz.alpha))
        # if self.update_step % self._cfg.actor_delay == 0:
        #     self.enta.update(enta)
        #     logger.logkv_mean('a_alpha', float(self.enta.alpha))
        # self.intrinsic_reward.update(rollout)


        self.update_step += 1
        if self.update_step % self._cfg.update_target_freq == 0:
            ema(self.nets, self.target_nets, self._cfg.tau)
            self.intrinsic_reward.ema(self._cfg.tau)