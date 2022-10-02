from tools.config import Configurable

class Relbo(Configurable):
    def __init__(self,
            info_net,
            prior_z,
            prior_a,
            cfg=None,
            ent_z=1.,
            ent_a=1.,
            mutual_info=1.,
            reward=1.,
            prior=None
     ) -> None:
        super().__init__(cfg)


    def __call__(self, traj):
        # always take a sequence as input
        raise NotImplementedError