# we now provide a modularized implementation of the RL training  
import torch
from typing import List
from .env_base import VecEnv
from .gae import HierarchicalGAE
from .traj import Trajectory
from .relbo import Relbo
from .ppo_agent import PPOAgent, CriticOptim
from .models import Policy, Critic, InfoNet
from .common_hooks import HookBase, RLAlgo, build_hooks
from tools.optim import TrainerBase


class RPG(RLAlgo):

    def __init__(
        self,
        env,
        p_z0,
        pi_a: PPOAgent,
        pi_z: PPOAgent,
        relbo: Relbo,
        gae: HierarchicalGAE,
        rnd=None,
        batch_size=256,

        rew_rms = None,
        obs_rms = None,
        hooks: List[HookBase] = []
    ) -> None:

        self.env = env
        self.pi_a = pi_a
        self.pi_z = pi_z
        self.p_z0 = p_z0
        self.relbo = relbo

        self.gae = gae
        self.latent_z = None


        self.rnd = rnd
        assert self.rnd is None
        self.batch_size = batch_size

        super().__init__(rew_rms, obs_rms, hooks)


    def modules(self):
        return [self.pi_a, self.pi_z, self.relbo, self.rew_rms, self.obs_rms]

    def inference(
        self,
        env: VecEnv,
        z,
        pi_z: PPOAgent,
        pi_a: PPOAgent,
        steps,
        **kwargs,
    ):
        transitions = []

        obs, timestep = self.start(env, **kwargs)
        if z is None:
            z = self.p_z0(len(obs))

        for step in range(steps):
            #old_z = z
            transition = dict(obs = obs, old_z = z, timestep = timestep)

            p_z = pi_z(obs, z, timestep=timestep)
            z, log_p_z = self.sample(p_z) # sample low-level z

            p_a = pi_a(obs, z, timestep=timestep)
            a, log_p_a = self.sample(p_a)

            assert log_p_a.shape == log_p_z.shape

            data, obs = self.step(env, a)


            transition.update(
                **data,
                a=a, z=z, log_p_a = log_p_a, log_p_z = log_p_z,
            )

            transitions.append(transition)
            timestep = transition['timestep']

            dones = transition['done']
            if dones.any():
                z = z.clone() #TODO: chance to dclone
                for idx, done in enumerate(dones):
                    if done:
                        z[idx] = self.p_z0()


        return Trajectory(transitions, len(obs), steps), z.detach()


    def evaluate(self, env, steps):
        with torch.no_grad():
            traj, _ = self.inference(env, None, self.pi_z, self.pi_a, steps=steps, reset=True)
            env.start(reset=True) # reset the force brutely
            return traj

    def run_rpg(self, env, steps):
        batch_size = self.batch_size
        with torch.no_grad():
            traj, self.latent_z = self.inference(
                env, self.latent_z, self.pi_z, self.pi_a, steps=steps)

            reward = self.relbo(traj, batch_size=batch_size)
            adv_targets = self.gae(traj, reward, batch_size=batch_size, rew_rms=self.rew_rms)

            data = traj.get_list_by_keys(['obs', 'timestep', 'a', 'old_z', 'z', 'log_p_a', 'log_p_z'])
            data.update(adv_targets)


        self.pi_a.learn(data, batch_size, ['obs', 'z', 'timestep', 'a', 'log_p_a', 'adv_a', 'vtarg_a'], logger_scope='pi_a')
        self.pi_z.learn(data, batch_size, ['obs', 'old_z', 'timestep', 'z', 'log_p_z', 'adv_z', 'vtarg_z'], logger_scope='pi_z')
        self.relbo.learn(data, batch_size) # maybe training with a seq2seq model way ..

        self.call_hooks(locals())



class train_rpg(TrainerBase):
    def __init__(self,
                    env: VecEnv,
                    hidden_space, # hidden space
                    cfg=None, steps=2048, device='cuda:0',
                    actor=Policy.gdc(
                        head=dict(TYPE='Normal',
                        std_mode='fix_learnable', std_scale=0.5)
                    ),
                    K = 0,
                    hidden_head=None,
                    critic=None,
                    ppo = PPOAgent.dc, #TODO: add adamW
                    gae = HierarchicalGAE.dc,
                    relbo = Relbo.dc,
                    info_net=InfoNet.to_build(TYPE='InfoNet'),
                    reward_norm=True,
                    initial_latent = 'zero',

                    obs_norm=False,
                    hooks = None,
                ):
        super().__init__()

        hidden_head = self.config_hidden(hidden_head, hidden_space)

        # set the default config
        # TODO: move it later ..
        if critic is None:
            critic = dict(backbone=actor.backbone)
        if not hasattr(info_net, 'backbone') or info_net.backbone is None:
            info_net.defrost()
            info_net.backbone = actor.backbone # shre te backbone if not specified


        from tools.utils import RunningMeanStd
        rew_rms = RunningMeanStd(last_dim=False) if reward_norm else None
        obs_rms = RunningMeanStd(clip_max=10.) if obs_norm else None

        obs_space = env.observation_space
        action_space = env.action_space
        
        # two policies

        actor_a = Policy(obs_space, hidden_space, action_space, cfg=actor).to(device)
        actor_z = Policy(obs_space, hidden_space, hidden_space, backbone=actor.backbone, head=hidden_head, K=K).to(device)

        critic_a = Critic(obs_space, hidden_space, env.reward_dim, cfg=critic).to(device)
        critic_z = Critic(obs_space, hidden_space, env.reward_dim, cfg=critic).to(device)

        pi_a = PPOAgent(actor_a, critic_a, cfg=ppo)
        pi_z = PPOAgent(actor_z, critic_z, cfg=ppo)


        info_net = InfoNet.build(obs_space, action_space, hidden_space, cfg=info_net)
        self.relbo = Relbo(info_net, hidden_space, action_space, cfg=relbo)


        hgae = HierarchicalGAE(pi_a, pi_z, cfg=gae)

        self.ppo = RPG(env, self.sample_initial_latent, pi_a, pi_z, self.relbo, hgae, rew_rms=rew_rms, obs_rms=obs_rms, hooks=build_hooks(hooks))

        while True:
            self.ppo.run_rpg(env, steps)


    def sample_initial_latent(self, size=1):
        z = torch.tensor([self.relbo.prior_z.sample()[0] for i in range(size)]).to('cuda:0')
        if self._cfg.initial_latent == 'zero':
            z = z * 0
        return z

    def config_hidden(self, hidden, hidden_space):
        from gym.spaces import Box, Discrete as Categorical
        from tools.config import merge_inputs, CN

        if isinstance(hidden_space, Box):
            default_hidden_head = dict(TYPE='Normal', linear=True, std_scale=0.2)
        elif isinstance(hidden_space, Categorical):
            default_hidden_head = dict(TYPE='Discrete')
        else:
            raise NotImplementedError

        if hidden is not None:
            hidden_head = merge_inputs(CN(default_hidden_head), **hidden)
        else:
            hidden_head = default_hidden_head
        return hidden_head