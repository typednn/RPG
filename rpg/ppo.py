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
        batch_size=256
    ) -> None:

        self.env = env
        self.pi = pi

        self.gae = gae,
        self.latent_z = None

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
            transition = dict(obs = obs,  timestep = timestep)

            p_a = pi(obs, None, timestep=timestep) # no z
            a, log_p_a = p_a.rsample()

            transition.update(
                **env.step(a),
                a=a,
                log_p_a = log_p_a,
                timestep = timestep.copy(),
                z=None,
            )
            obs = transitions.pop('obs')
            timestep = transition['timestep']
            transitions.append(transition)

        return Trajectory(transitions, len(obs), steps)


    def run_ppo(self, env, steps):
        with torch.no_grad():
            traj = self.inference(env, self.latent_z, self.pi, steps=steps)
            adv_targets = self.gae(traj, traj.get_tensor(['reward']), batch_size=self._cfg.batch_size)
            key = ['obs', 'a', 'log_p_a']
            data = {traj.get_list(key) for key in key}
            data.update(adv_targets)
            data['z'] = None

        self.pi.learn(data, self._cfg.batch_size, ['obs', 'z', 'timestep', 'a', 'log_p_a', 'adv', 'vtarg'])



class train_ppo(TrainerBase):
    def __init__(self, env: VecEnv, cfg=None, steps=2048):
        super().__init__()
        obs_space = env.observation_space
        action_space = env.action_space
        actor = Policy(obs_space, None, action_space)
        critic = Critic(obs_space, None, 1)
        pi = PPOAgent(actor, critic, GAE())
        self.ppo = PPO(env, pi, GAE())
        self.ppo.run_ppo(env, steps)
