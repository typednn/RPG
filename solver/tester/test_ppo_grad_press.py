import os
from cpt.solver.networks import PointnetBackbone
from cpt.solver import make_env
from diffrl.utils import logger
from diffrl.rl.vec_envs import SubprocVectorEnv, DummyVectorEnv
from diffrl.rl.agents.ppo_agent import PPOAgent
os.environ["OMP_NUM_THREADS"] = "1"  # it can accelerate the training?

logger.configure('/tmp/ppo_grad_press')

if __name__ == '__main__':
    from torch.multiprocessing import set_start_method
    set_start_method('spawn')
    env = SubprocVectorEnv([lambda: make_env('grad_press') for i in range(1)])
    #env = DummyVectorEnv([lambda: make_env('press') for i in range(2)])
    agent = PPOAgent.parse(
        env.observation_space[0], env.action_space[0],
        nsteps=20, roll_times=10, eval_episode=5, show_roller_progress=True, batch_size=50, ppo_optim=dict(max_kl=0.1),
        evaluator_cfg=dict(render_episodes=1),
        actor=dict(backbone=dict(TYPE="PointnetBackbone"), head=dict(TYPE="MaskHead", std_scale=0.1)),
        obs_norm=False, compute_gradient=True, value_gradient=True, n_epochs=5, reward_norm=False
    ).cuda()

    print('start ...')
    for i in range(1000000):
        agent.train(env)
