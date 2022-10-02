# we now provide a modularized implementation of the RL training  
import torch
from .env_base import VecEnv
from .model_base import Policy
from .gae import ComputeTargetValues
from .traj import Trajectory
from .relbo import Relbo
from .utils import minibatch_gen
from tools.utils import totensor


class Trainer:

    def __init__(
        self,
        env,
        pi_a,
        pi_z,
        critic,
        relbo: Relbo,

        gae: ComputeTargetValues,
        rew_rms=None,
    ) -> None:

        self.on_policy = False
        self.off_policy = False

        self.training = True
        self.gae = gae,
        self.latent_z = None

    def add(self, plugin):
        self.plugins.append(plugin)


    def inference(
        self,
        env: VecEnv,
        z,
        pi_z: Policy,
        pi_a: Policy,
        steps,
        init_z_fn = lambda x: 0,
        **kwargs,
    ):
        transitions = []
        obs, timestep = env.start(**kwargs)

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
                z[dones] = init_z_fn(dones) #reset z

        return Trajectory(transitions, len(obs), steps), z


    def main_loop(self, **kwargs):
        batch_size = 128

        with torch.no_grad():
            traj, z = self.inference(**kwargs)

            # pre-training
            reward = self.relbo(traj)
            adv_targets = self.gae(traj, reward, batch_size=batch_size)

            key = ['obs', 'a', 'old_z', 'z', 'log_p_a', 'log_p_z',]
            data = {traj.get_list(key) for key in key}
            data.update(adv_targets)

            key += ['adv_a', 'adv_z', 'vtarg_a', 'vtarg_z']

            for i in range(5):
                # only update one episode ..
                for batch in minibatch_gen(data, batch_size):
                    self.pi_a.step(
                        batch['obs'], batch['z'], batch['a'], batch['log_p_a'], batch['adv_a'])
                    self.pi_z.step(
                        batch['obs'], batch['old_z'], batch['z'], batch['log_p_z'], batch['adv_z'])
                    self.critic.step(
                        batch['obs'], batch['old_z'], batch['z'], batch['vtarg_a'], batch['vtarg_z']
                    )
                self.relbo.step(traj)