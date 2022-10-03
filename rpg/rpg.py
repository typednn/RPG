# we now provide a modularized implementation of the RL training  
import torch
from .env_base import VecEnv
from .gae import HierarchicalGAE
from .traj import Trajectory
from .relbo import Relbo
from .ppo_agent import PPOAgent, CriticOptim
from .models import Policy, Critic, InfoNet
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
        batch_size=2000,
    ) -> None:

        self.env = env
        self.pi_a = pi_a
        self.pi_z = pi_z
        self.p_z0 = p_z0
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
                a=a, z=z, log_p_a = log_p_a, log_p_z = log_p_z,
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


    def run_rpg(self, env, steps):
        batch_size = self.batch_size
        with torch.no_grad():
            traj, self.latent_z = self.inference(
                env, self.latent_z, self.pi_z, self.pi_a, steps=steps)

            reward = self.relbo(traj, batch_size=batch_size)
            adv_targets = self.gae(traj, reward, batch_size=batch_size)

            data = {traj.get_list(key) for key in ['obs', 'a', 'old_z', 'z', 'log_p_a', 'log_p_z']}
            data.update(adv_targets)


        self.pi_a.learn(data, batch_size, ['obs', 'z', 'timestep', 'a', 'log_p_a', 'adv_a', 'vtarg_a'])
        self.pi_z.learn(data, batch_size, ['obs', 'old_z', 'timestep', 'z', 'log_p_z', 'adv_z', 'vtarg_z'])
        self.relbo.learn(data) # maybe training with a seq2seq model way ..



class train_rpg(TrainerBase):
    def __init__(self,
                    env: VecEnv,
                    hidden_space, # hidden space
                    cfg=None, steps=2048, device='cuda:0',
                    actor=Policy.gdc(
                        head=dict(TYPE='Normal',
                        std_mode='fix_learnable', std_scale=0.5)
                    ),
                    hidden=Policy.dc,
                    critic=Critic.dc,
                    ppo = PPOAgent.dc,
                    gae = HierarchicalGAE.dc,
                    relbo = Relbo.dc,
                    info_net=InfoNet.to_build(TYPE='InfoNet'),
                    reward_norm=True,
                    initial_latent = 'zero'
                ):
        super().__init__()


        from tools.utils import RunningMeanStd
        rew_rms = RunningMeanStd(last_dim=False) if reward_norm else None

        obs_space = env.observation_space
        action_space = env.action_space
        
        # two policies

        actor_a = Policy(obs_space, hidden_space, action_space, cfg=actor).to(device)
        actor_z = Policy(obs_space, hidden_space, hidden_space, cfg=hidden).to(device)

        critic_a = Critic(obs_space, hidden_space, env.reward_dim, cfg=critic).to(device)
        critic_z = Critic(obs_space, hidden_space, env.reward_dim, cfg=critic).to(device)

        pi_a = PPOAgent(actor_a, critic_a, cfg=ppo)
        pi_z = PPOAgent(actor_z, critic_z, cfg=ppo)


        info_net = InfoNet.build(obs_space, action_space, hidden_space, cfg=info_net)
        self.relbo = Relbo(info_net, hidden_space, action_space, relbo=relbo)

        #p_z0
        def sample_latent(size=1):
            #[for i in range(size)]
            z = self.relbo.prior_z.sample(size)
            if initial_latent == 'zero':
                z = z * 0
            return z

        hgae = HierarchicalGAE(pi_a, pi_z, cfg=gae)

        self.ppo = RPG(env, sample_latent, pi_a, pi_z, hgae, self.relbo, rew_rms=rew_rms)

        while True:
            self.ppo.run_rpg(env, steps)
