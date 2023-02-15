from omegaconf import OmegaConf as C


class ConfigurableBase:
    @classmethod
    def default_config(cls) -> C:
        return C.create()