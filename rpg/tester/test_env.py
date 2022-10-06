import tqdm
import numpy as np
from rpg.env_base import GymVecEnv, TorchEnv


def test_gym():
    N = 3
    env = GymVecEnv('HalfCheetah-v3', N)

    obs, timestep = env.start()
    for i in tqdm.trange(10000):
        output = env.step([env.action_space.sample() for i in range(env.nenv)])
        if i % 1000 == 999:
            for i in output['done']:
                assert i
            for i in output['timestep']:
                assert i == 0, f"timestep is {i}"
            assert len(output['episode']) == N
            for a, b in zip(output['obs'], output['next_obs']):
                assert not np.allclose(a, b)

        if i % 1000 == 998:
            for i in output['done']:
                assert not i
            for i in output['timestep']:
                assert i == 999, f"timestep is {i}"
            for a, b in zip(output['obs'], output['next_obs']):
                assert np.allclose(a, b)
            assert len(output['episode']) == 0


def test_tripush():
    from tools.utils import totensor 
    N = 100
    env = TorchEnv('TripleMove', N)
    obs, timestep = env.start()
    print(obs.shape, timestep.shape, obs.dtype, timestep.dtype)
    for i in tqdm.trange(1000):
        action = totensor([env.action_space.sample() for i in range(env.nenv)], 'cuda:0')
        transition = env.step(action)


def test_maze():
    from tools.utils import totensor, animate
    N = 100
    env = TorchEnv('Maze', N)
    obs, timestep = env.start()

    images = []

    for i in tqdm.trange(200):
        action = totensor([env.action_space.sample() for i in range(env.nenv)], 'cuda:0')
        env.step(action)
        images.append(env.render('rgb_array'))
    
    print(len(images))
    animate(images, 'animation.mp4')
    


if __name__ == '__main__':
    #test_gym()
    #test_tripush()
    test_maze()