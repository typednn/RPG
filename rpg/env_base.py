from abc import ABC, abstractmethod


class VecEnv(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def start(self):
        # similar to reset, but return obs [batch of observations: nunmy array/dict/tensors] and timesteps [numpy].
        # if done or no obs, reset it.
        # note that this process is never differentiable.
        pass

    @abstractmethod
    def step(self, action):
        # return {'next_obs': _, 'new_obs', 'reward': reward, 'done': done, 'info': info}
        # next obs will not be obs if done.
        pass