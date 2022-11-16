import tqdm
from envs.softbody.plb_envs import RopeEnv
from tools.utils import animate

images = []
env = RopeEnv()
out = env.reset()
images.append(env.render('rgb_array', spp=1))
for i in tqdm.trange(100):
    out = env.step(env.action_space.sample())[0]
    images.append(env.render('rgb_array', spp=1))

animate(images, 'output.mp4', fps=30)