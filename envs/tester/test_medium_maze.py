import torch
import matplotlib.pyplot as plt
from envs.maze import MediumMaze, SmallMaze
from tools.utils import totensor
import tqdm
import matplotlib.pyplot as plt

N = 1000
#env = MediumMaze(batch_size=N, reward_mapping=[[[6, 1], 10], [[6, 6], 20]])
env = SmallMaze(batch_size=N)

trajs = []

obs = env.reset()


image = env.render('rgb_array')
plt.imshow(image)
plt.savefig('test.png')

# trajs.append(obs)

# for i in tqdm.trange(10000):
#     #env.step()
#     actions = totensor([env.action_space.sample() for i in range(N)], device='cuda:0')
#     obs = env.step(actions)[0]
#     trajs.append(obs)
#     if (i+1) % 40 == 0:
#         obs = env.reset()
# print(len(trajs))

# data = env._render_traj_rgb({'next_obs': torch.cat(trajs)})

# import numpy as np

# background = data.get('background', {})

# if 'image' in background:
#     plt.imshow(np.uint8(background['image']*255))
# if 'xlim' in background:
#     plt.xlim(background['xlim'])
# if 'ylim' in background:
#     plt.ylim(background['ylim'])


# state = data['state']
# plt.scatter(state[..., 0], state[..., 1])
# #images[name] = plt_save_fig_array()[:, :, :3]
# plt.savefig('test.png')