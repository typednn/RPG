import numpy as np
import matplotlib.pyplot as plt
import tqdm
from envs.maniskill.fixed_arm import FixArm


env = FixArm()
env.reset()
print(env.action_space.shape)

img = env.render('rgb_array')
plt.imshow(img)
plt.savefig('arm.png')

images = []
trajs = []
history = None
for i in tqdm.trange(10000):
    obs = env.step(env.action_space.sample())[0]
    #images.append(env.render('rgb_array'))
    #print(env.agent.base_pose)
    trajs.append(obs)
    if (i + 1) % 100 == 0:
        data = env._render_traj_rgb({'next_obs': np.array(trajs)}, occ_val=1, history=history)
        history = data['history']

        print(data['metric']['occ'])
        images.append(data['image']['hist'])

        plt.clf()
        plt.imshow(data['image']['hist'])
        plt.savefig('output.png')

        trajs = []

from tools.utils import animate
animate(images, 'output.mp4')