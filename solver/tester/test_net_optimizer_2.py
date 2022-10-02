from collections import deque
from cpt.solver import task_sampler
from cpt.solver.iter_nn import Solver
from cpt.solver import make_env
from diffrl.utils import animate 


def main():
    env = make_env()
    solver = Solver.parse(env, low_steps=50, parse_prefix='solver')
    high_buffer = deque(maxlen=200)

    for i in range(10000):
        if i % 10 == 0:
            images = []
        else:
            images = None
        for i in range(5):
            start_state, goal = task_sampler.multi_goal(env)
            solver.sample_trajectory(start_state, goal, num_stages=1, num_grad_iter=0, images=images, high_buffer=high_buffer)

        if images is not None:
            animate(images, 'animate2.mp4')
        solver.optimize_low_actor(high_buffer, 50, num_stages=5)

if __name__ == '__main__':
    main()