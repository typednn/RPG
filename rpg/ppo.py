# an implementation of PPO to verify the correctness of env_base ..
import torch
from .env_base import VecEnv
from .models import Policy, Critic
from .gae import GAE
from .traj import Trajectory
from .ppo_agent import PPOAgent
from tools.optim import TrainerBase


class PPO:
    def __init__(
        self,
        env,
        pi: PPOAgent,
        gae: GAE,
        batch_size=256,
        rew_rms=None
    ) -> None:

        self.env = env
        self.pi = pi

        self.gae = gae
        self.latent_z = None
        self.batch_size = batch_size

        self.rew_rms = None

    def inference(
        self,
        env: VecEnv,
        pi: PPOAgent,
        steps,
        **kwargs,
    ):
        transitions = []
        obs, timestep = env.start(**kwargs)
        for step in range(steps):
            transition = dict(obs=obs, timestep = timestep)
            p_a = pi(obs, None, timestep=timestep) # no z
            a, log_p_a = p_a.rsample()
            data = env.step(a)
            obs = data.pop('obs')
            transition.update(
                **data,
                a=a,
                log_p_a = log_p_a,
                z=None,
            )
            timestep = transition['timestep']
            transitions.append(transition)

        return Trajectory(transitions, len(obs), steps)


    def run_ppo(self, env, steps):
        with torch.no_grad():
            traj = self.inference(env, self.pi, steps=steps)

            reward = traj.get_tensor('r'); assert reward.dim() == 3, "rewards must be (nstep, nenv, reward_dim)"
            adv_targets = self.gae(traj, reward, batch_size=self.batch_size, rew_rms=self.rew_rms, debug=False)

            data = traj.get_list_by_keys( ['obs', 'timestep', 'a', 'log_p_a'])
            data.update(adv_targets)
            data['z'] = None

        self.pi.learn(data, self.batch_size, ['obs', 'z', 'timestep', 'a', 'log_p_a', 'adv', 'vtarg'])
        print(traj.summarize_epsidoe_info())



class train_ppo(TrainerBase):
    def __init__(self,
                        env: VecEnv, cfg=None, steps=2048, device='cuda:0',
                        actor=Policy.get_default_config(
                            head=dict(std_mode='fix_learnable', std_scale=0.5)
                        ),
                        critic=Critic.get_default_config(),
                        ppo = PPOAgent.get_default_config(),
                        gae = GAE.get_default_config(),
                        reward_norm=True,
                ):
        super().__init__()


        from tools.utils import RunningMeanStd
        rew_rms = RunningMeanStd(last_dim=True) if reward_norm else None

        obs_space = env.observation_space
        action_space = env.action_space
        actor = Policy(obs_space, None, action_space, cfg=actor).to(device)
        critic = Critic(obs_space, None, 1, cfg=critic).to(device)
        pi = PPOAgent(actor, critic, ppo)
        self.ppo = PPO(env, pi, GAE(pi, cfg=gae), rew_rms=rew_rms)

        while True:
            self.ppo.run_ppo(env, steps)
