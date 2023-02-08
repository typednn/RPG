## Generator

Code base for generative models

### Dataset
- defines the 
    - batch loader
    - type of the input samples
    - type of the output samples

### Encoder
backbone of various types:
- MLP
- CNN
- RNN
- Transformer

By default for Encoder(TypeA, TypeB) we will try to process TypeA to handle TypeB

### models
probablistic models and training utilities
- VAE
- GAN
- Diffusion

being composed of different
- loss types as utils


trainer will be defined in the main.py, providing a class that
- type annotation for training and inference
- can be called for inference given inputs
- or be called for training and compute the losses.