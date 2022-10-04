# Variational Reparametrized Policy Learning with Differentiable Physics

Model-free and model-based learning with Reparameterized Policy Learning (RPG).

## Architecture

- `rpg`: the major  algorithms for RPG
- `tools`: utilities for training and testing, including configuration, logging, and visualization 
- `nn`: neural network modules
    - `nn/spaces`: determine the input structure of the networks
    - `nn/distributions`: determine the output structure of the networks
    - `nn/modules`: backbones, supports both MLP, CNN, PointNet and Transformers
- `rl`: RL baselines and common RL utilities
- `envs`: environments
- `generative`: generative models for reinforcement leraning, including 
    - dynamic learning like model-based RL
    - diffusion model for trajectory modeling
    - RND for density estimation for exploration