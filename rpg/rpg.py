# we now provide a modularized implementation of the RL training  
import torch
from .env_base import VecEnv
from .model_base import Policy
from .plugin import Plugin
from .utils import predict_traj_value, convert_traj_key_to_tensor
from tools.utils import dstack, totensor



class Relbo(Plugin):
    def __init__(self, info_net,
            cfg=None,
            ent_z=1.,
            ent_a=1.,
            mutual_info=1.,
            reward=1.,
            prior=None
     ) -> None:
        super().__init__(cfg)


class ComputeTargetValues(Plugin):
    # @ litian, write the unit test code for the target value computation..
    def __init__(self, critic_a, critic_z, cfg=None, gamma=0.995, lmbda=0.97):
        super().__init__(cfg)
        self.critic_a = critic_a
        self.critic_z = critic_z


    @torch.no_grad()
    def update_data(self, data, locals_):
        # must determin the batch size..
        batch_size = locals_['batch_size']
        rew_rms = locals_.get('rew_rms', None)

        traj = data['traj']
        nenv = traj['nenv']
        timesteps = traj['timesteps']

        ind = []
        for j in range(timesteps):
            for i in range(nenv):
                if traj[j]['done'][i] or j == timesteps -1:
                    ind.append(j, i)
        ind = totensor(ind, device=torch.long)

        vpredz = predict_traj_value(traj, ('obs', 'old_z', 'timestep'), self.critic_z, batch_size=batch_size)
        vpreda = predict_traj_value(traj, ('obs', 'z', 'timestep'), self.critic_a, batch_size=batch_size)
        next_vpredz = predict_traj_value(traj, ('next_obs', 'z', 'timestep'), self.critic_z, batch_size=batch_size)
        next_vpreda = torch.zeros_like(next_vpredz)
        next_vpreda[ind] = next_vpredz[ind] # there is no action estimate .. 

        reward = convert_traj_key_to_tensor(traj, 'r') # compute trajectorye
        done = convert_traj_key_to_tensor(traj, 'done')

        lmbda_sqrt = self._cfg.lmbda**0.5

        vpred = vpredz + vpreda * lmbda_sqrt
        next_vpred = vpredz * vpreda * lmbda_sqrt

        
        adv = torch.zeros_like(next_vpred)
        for t in reversed(len(vpredz)):
            nextvalue = next_vpred[t]
            mask = 1. - done[t]
            assert mask.shape == nextvalue.shape
            #print(reward.device, next_vpred.device, mask.device, vpred[t].device)
            delta = reward[t] + self._cfg.gamma * nextvalue * mask - vpred[t]
            adv[t] = lastgaelam = delta + self._cfg.gamma * self._cfg.lmbda * lastgaelam * mask
        vtarg = vpred + adv

        traj['adv_a'] = vtarg - vpreda
        traj['vtarget_a'] = vtarg

        vtarg_z = vpreda + lmbda_sqrt * vtarg 
        traj['adv_z'] = vtarg_z - vpredz
        traj['vtarget_z'] = vtarg_z


    def comptue_gae(self, reward, vpred, next_vpred, done):
        lastgaelam = 0
        adv = torch.zeros_like(vpred)
        for t in reversed(len(vpred)):
            nextvalue = next_vpred[t]
            mask = 1. - done[t]
            assert mask.shape == nextvalue.shape
            #print(reward.device, next_vpred.device, mask.device, vpred[t].device)
            delta = reward[t] + self._cfg.gamma * nextvalue * mask - vpred[t]
            adv[t] = lastgaelam = delta + self._cfg.gamma * self._cfg.lmbda * lastgaelam * mask
            # nextvalue = vpred[t]
        vtarg = vpred + adv
        return adv, vtarg


class Roller:

    def __init__(
        self,
    ) -> None:

        self.on_policy = False
        self.off_policy = False

        self.training = True

        self.plugins = [ComputeTargetValues()]

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
            obs = transitions.pop('new_obs')

            transitions.append(transition)

            timestep = timestep + 1

            dones  = dstack(transition['done'])
            if dones.any():
                z = z.clone()
                z[dones] = init_z_fn(dones) #reset z

        transitions['timesteps'] = steps
        transitions['nenv'] = len(obs)
        return transitions


    def main_loop(self, **kwargs):
        data = {
            'traj': self.inference(**kwargs),
        }

        # pre-training
        for i in self.plugins:
            data = i._update_data(data, locals())