from setuptools import setup

install_requires = [
    'scipy',
    'numpy',
    'wandb',
    'torch',
    'tensorboardX',
    'opencv-python',
    'tqdm',
    'taichi',
    'gym==0.25.2',
    'tensorboard',
    'yacs>=0.1.8',
    'matplotlib',
    'descartes',
    #'shapely',
    #'natsort',
    'torchvision',
    'einops',
    #'alphashape',
    'transforms3d',
    'h5py',
    #'bezier',
    'pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git',
    'chamferdist',
    #'geomloss',
    'open3d',
    'pydprint',
    'pyro-ppl',
    'moviepy>=1.0.3',
    'gitpython',
    'ninja',
    'diffusers',
    #'pyvista', # maybe not needed
    #'pythreejs', # maybe not needed
]


setup(
    name='rpg',
    version='0.0.1',
    install_requires=install_requires,
    py_modules=['rl', 'tools', 'solver']
)

# for RTX 30 series card, install pytorch with cudatoolkit 11 to support sm_86
# conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
