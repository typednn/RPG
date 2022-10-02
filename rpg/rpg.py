# we now provide a modularized implementation of the RL training  
import torch
from .env_base import VecEnv
from .model_base import Policy
from .gae import ComputeTargetValues
from .traj import Trajectory
from .relbo import Relbo
from .utils import minibatch_gen
from .ppo_actor import PPO
from .critic import Critic
from tools.utils import totensor


class Trainer:

    def __init__(
        self,
        env,
        pi_a: PPO,
        pi_z: PPO,
        critic,
        relbo: Relbo,

        gae: ComputeTargetValues,

        rew_rms=None,
        rnd=None,
        batch_size=None
    ) -> None:

        self.env = env
        self.pi_a = pi_a
        self.pi_z = pi_z
        self.critic = critic
        self.relbo = relbo

        self.gae = gae,
        self.latent_z = None

        self.rew_rms = rew_rms
        self.rnd = rnd

    def inference(
        self,
        env: VecEnv,
        z,
        pi_z: Policy,
        pi_a: Policy,
        steps,
        **kwargs,
    ):
        transitions = []
        obs, timestep = env.start(**kwargs)
        if z is None:
            self.pi_z.zero(obs)
            raise NotImplementedError

        for step in range(steps):
            #old_z = z
            transition = dict(obs = obs, old_z = z, timestep = timestep)

            p_z = pi_z(obs, z, timestep=timestep)
            z, log_p_z = p_z.sample() # sample low-level z

            p_a = pi_a(obs, z, timestep=timestep)
            a, log_p_a = p_a.rsample()

            assert log_p_a.shape == log_p_z.shape

            transition.update(env.step(a))
            transition.update(
                a=a,
                z=z,
                log_p_a = log_p_a,
                log_p_z = log_p_z,
            )
            obs = transitions.pop('new_obs')

            transitions.append(transition)

            timestep = timestep + 1

            dones  = totensor(transition['done'])
            if dones.any():
                z = z.clone()
                z[dones] = self.pi_z.zero(obs[dones]) #reset z

        return Trajectory(transitions, len(obs), steps), z


    def run_ppo(self, env, steps):
        batch_size = self._cfg.batch_size or 128

        with torch.no_grad():
            traj, self.latent_z = self.inference(
                env, self.latent_z, self.pi_z, self.pi_a, steps=steps)

            # pre-training
            reward = self.relbo(traj)
            adv_targets = self.gae(traj, reward, batch_size=batch_size)

            key = ['obs', 'a', 'old_z', 'z', 'log_p_a', 'log_p_z']
            data = {traj.get_list(key) for key in key}
            data.update(adv_targets)

            key += ['adv_a', 'adv_z', 'vtarg_a', 'vtarg_z']
            stop_a = False
            stop_z = False

            for i in range(5):
                # only update one episode ..
                for batch in minibatch_gen(data, batch_size):
                    if not stop_a:
                        stop_a = self.pi_a.step(
                            batch['obs'], batch['z'], batch['a'], batch['log_p_a'], batch['adv_a']
                        )

                    if not stop_z:
                        stop_z = self.pi_z.step(
                            batch['obs'], batch['old_z'], batch['z'], batch['log_p_z'], batch['adv_z']
                        )

                    self.critic.step(
                        batch['obs'], batch['old_z'], batch['z'], batch['vtarg_a'], batch['vtarg_z']
                    )

            self.relbo.train(traj) # maybe training with a seq2seq model way ..
