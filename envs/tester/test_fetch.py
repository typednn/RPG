from multiprocessing import Pool


def work(arg):
    from envs.fetch.sequential import SequentialStack
    env = SequentialStack()
    env.reset()
    images = []
    for i in range(3):
        env.reset()
        images.append(env.render('rgb_array'))
        for j in range(100):
            env.step(env.action_space.sample())
            images.append(env.render('rgb_array'))
    return images

with Pool(1) as p:
    for i in p.map(work, range(1)):
        images = i

from tools.utils import animate
animate(images, 'output.mp4')