import tqdm
import numpy as np
from rpg.env_base import GymVecEnv


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



if __name__ == '__main__':
    test_gym()