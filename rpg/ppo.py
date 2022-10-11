# an implementation of PPO to verify the correctness of env_base ..
import torch
from typing import Optional
from .env_base import VecEnv
from .models import Policy, Critic
from .rnd import RNDOptim
from .traj import Trajectory
from .ppo_agent import PPOAgent
from tools.optim import TrainerBase
from typing import List
from .common_hooks import HookBase, build_hooks, RLAlgo
from tools.utils import RunningMeanStd


class PPO(RLAlgo):
    def __init__(
        self,
        env,
        pi: PPOAgent,
        batch_size=256,

        rnd: Optional[RNDOptim] = None,

        obs_rms=None,
        hooks: List[HookBase] = [],
    ) -> None:

        self.env = env
        self.pi = pi
        self.latent_z = None
        self.batch_size = batch_size
        self.rnd = rnd

        super().__init__(obs_rms, hooks)


    def modules(self):
        return [self.pi, self.obs_rms]

    def inference(
        self,
        env: VecEnv,
        pi: PPOAgent,
        steps,
        **kwargs,
    ):
        transitions = []
        obs, timestep = self.start(env, **kwargs)

        for step in range(steps):
            transition = dict(obs=obs, timestep = timestep)
            p_a = pi(obs, None, timestep=timestep) # no z
            a, log_p_a = self.sample(p_a)
            data, obs = self.step(env, a)
            transition.update(
                **data,
                a=a,
                log_p_a = log_p_a,
                z=None,
            )
            timestep = transition['timestep']
            transitions.append(transition)

        return Trajectory(transitions, len(obs), steps)

    def evaluate(self, env, steps):
        with torch.no_grad():
            traj = self.inference(env, self.pi, steps=steps, reset=True)
            env.start(reset=True) # reset the force brutely
            return traj

    def run_ppo(self, env, steps):
        with torch.no_grad():
            traj = self.inference(env, self.pi, steps=steps)

            reward = traj.get_tensor('r', device='cuda:0')
            logp = traj.get_tensor('log_p_a', device='cuda:0')

            if self.rnd is not None:
                rnd_reward = self.rnd(traj, batch_size=self.batch_size, update_normalizer=True)
                reward = torch.cat((reward, rnd_reward), dim=-1) # 2 dim rewards ..
                
            done, truncated = traj.get_truncated_done()
            vpred = traj.predict_value(('obs', 'z', 'timestep'), self.pi.value, batch_size=self.batch_size)
            #TODO: timestep to the next timestep
            next_vpred = traj.predict_value(('next_obs', 'z', 'timestep'), self.pi.value, batch_size=self.batch_size)
            adv_targets = self.pi.gae(vpred, next_vpred, reward, done, truncated, logp=logp)


            data = traj.get_list_by_keys(['obs', 'timestep', 'a', 'log_p_a'])
            data.update(adv_targets)
            data['z'] = None

        self.pi.learn(data, self.batch_size, ['obs', 'z', 'timestep', 'a', 'log_p_a', 'adv', 'vtarg'], logger_scope='')
        if self.rnd is not None:
            self.rnd.learn(data, batch_size=self.batch_size, logger_scope='rnd')

        self.call_hooks(locals())



class train_ppo(TrainerBase):
    def __init__(
                self,
                env: VecEnv, cfg=None, steps=2048, device='cuda:0',
                actor=Policy.get_default_config(
                    head=dict(std_mode='fix_learnable', std_scale=0.5)
                ),
                critic=Critic.dc,
                ppo = PPOAgent.dc,
                obs_norm=True,
                batch_size=256,
                hooks = None,
                rnd=None,
        ):
        #super().__init__()
        TrainerBase.__init__(self)

        obs_rms = RunningMeanStd(clip_max=10.) if obs_norm else None

        obs_space = env.observation_space
        action_space = env.action_space
        actor = Policy(obs_space, None, action_space, cfg=actor).to(device)
        critic = Critic(
            obs_space, None,
            env.reward_dim + (rnd is not None) + (ppo.entropy.coef > 0.),
            cfg=critic
        ).to(device)

        pi = PPOAgent(actor, critic, ppo)

        if rnd is not None:
            rnd = RNDOptim(obs_space, cfg=rnd).to(device)
        self.ppo = PPO(
            env, pi,
            obs_rms=obs_rms,
            batch_size=batch_size,
            hooks=build_hooks(hooks),
            rnd=rnd,
        )

        while True:
            self.ppo.run_ppo(env, steps)
