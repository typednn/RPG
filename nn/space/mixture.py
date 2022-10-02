from ..distributions import ActionDistr, CategoricalAction, DistHead, NormalAction, MixtureAction

class MixtureSpace:
    def __init__(self, discrete, continuous) -> None:
        self.continuous = continuous
        self.discrete = discrete