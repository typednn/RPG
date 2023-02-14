from omegaconf import OmegaConf as C

class ConfigurableBase:
    @classmethod
    def default_config(cls) -> C:
        return C.create()

class Configurable(ConfigurableBase):
    def __init__(self) -> None: 
        self._init_kwargs = C.create()

    @property
    def config(self):
        if not hasattr(self, '_config') or self._config is None:
            self.build_config()
        return self._config

    @property
    def pretty_config(self):
        return C.to_yaml(self.config)

    @classmethod
    def _new_config(cls)->C:
        return C.create()

    @classmethod
    def default_config(cls) -> C:
        return C.merge(
            super().default_config(),
            cls._new_config()
        )

    def build_config(self) -> C:
        # build config from self._init_kwargs
        self._config = C.merge(self.default_config(), self._init_kwargs)