# we now provide a modularized implementation of the RL training  
import torch
from .env_base import VecEnv
from .gae import HierarchicalGAE
from .traj import Trajectory
from .relbo import Relbo
from .ppo_agent import PPOAgent, CriticOptim
from .models import Policy, Critic
from tools.utils import totensor
from tools.optim import TrainerBase


class RPG:

    def __init__(
        self,
        env,
        p_z0,
        pi_a: PPOAgent,
        pi_z: PPOAgent,
        relbo: Relbo,
        gae: HierarchicalGAE,
        rew_rms=None,
        rnd=None,
        batch_size=256
    ) -> None:

        self.env = env
        self.pi_a = pi_a
        self.pi_z = pi_z
        self.relbo = relbo

        self.gae = gae,
        self.latent_z = None

        self.rew_rms = rew_rms
        self.rnd = rnd
        self.batch_size = batch_size


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
        obs, timestep = env.start(**kwargs)
        if z is None:
            z = self.p_z0.sample(len(obs))

        for step in range(steps):
            #old_z = z
            transition = dict(obs = obs, old_z = z, timestep = timestep)

            p_z = pi_z(obs, z, timestep=timestep)
            z, log_p_z = p_z.sample() # sample low-level z

            p_a = pi_a(obs, z, timestep=timestep)
            a, log_p_a = p_a.rsample()

            assert log_p_a.shape == log_p_z.shape


            data = env.step(a)
            obs = data.pop('obs')
            transition.update(
                **data,
                a=a,
                z=z,
                log_p_a = log_p_a,
                log_p_z = log_p_z,
            )

            transitions.append(transition)
            timestep = transition['timestep']

            dones  = totensor(transition['done'])
            if dones.any():
                z = z.clone() #TODO: chance to dclone
                for idx, done in enumerate(dones):
                    if done:
                        z[idx] = self.p_z0.sample()


        return Trajectory(transitions, len(obs), steps), z


    def run_ppo(self, env, steps):
        with torch.no_grad():
            traj, self.latent_z = self.inference(
                env, self.latent_z, self.pi_z, self.pi_a, steps=steps)

            reward = self.relbo(traj)
            adv_targets = self.gae(traj, reward, batch_size=self._cfg.batch_size)

            data = {traj.get_list(key) for key in ['obs', 'a', 'old_z', 'z', 'log_p_a', 'log_p_z']}
            data.update(adv_targets)


        self.pi_a.learn(data, self.batch_size, ['obs', 'z', 'timestep', 'a', 'log_p_a', 'adv_a', 'vtarg_a'])
        self.pi_z.learn(data, self.batch_size, ['obs', 'old_z', 'timestep', 'z', 'log_p_z', 'adv_z', 'vtarg_z'])
        self.relbo.learn(data) # maybe training with a seq2seq model way ..



class train_rpg(TrainerBase):
    def __init__(self,
                        env: VecEnv, cfg=None, steps=2048, device='cuda:0',
                        actor=Policy.get_default_config(
                            head=dict(std_mode='fix_learnable', std_scale=0.5)
                        ),
                        critic=Critic.get_default_config(),

                        ppo = PPOAgent.get_default_config(),
                        gae = HierarchicalGAE.get_default_config(),
                        reward_norm=True,
                ):
        super().__init__()


        from tools.utils import RunningMeanStd
        rew_rms = RunningMeanStd(last_dim=False) if reward_norm else None

        obs_space = env.observation_space
        action_space = env.action_space
        actor = Policy(obs_space, None, action_space, cfg=actor).to(device)
        critic = Critic(obs_space, None, 1, cfg=critic).to(device)
        pi = PPOAgent(actor, critic, ppo)
        self.ppo = RPG(env, pi, HierarchicalGAE(pi, cfg=gae), rew_rms=rew_rms)

        while True:
            self.ppo.run_ppo(env, steps)
