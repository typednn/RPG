from cpt.solver import task_sampler
from cpt.solver.iter_nn import Solver
from cpt.solver import make_env
from diffrl.utils import animate 


def main():
    env = make_env()
    start_state, goal = task_sampler.box_press(env)
    env.reset_state_goal(start_state, goal)
    solver = Solver.parse(env, low_steps=50, parse_prefix='solver')

    images = []
    solver.sample_trajectory(start_state, goal, 2, num_grad_iter=200, images=images)
    animate(images)

if __name__ == '__main__':
    main()