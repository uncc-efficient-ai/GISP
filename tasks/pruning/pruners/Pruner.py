from abc import ABC, abstractmethod
import modules.system.system as system


class Pruner(ABC):
    @abstractmethod
    def __init__(self, model, config, data):
        self.model = model
        self.config = config
        self.data = data
        self.is_peft = config.model.model_peft

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_imps(self):
        pass

    def get_model(self):
        if system.enable_deepspeed:
            return self.model.module
        elif self.is_peft:
            return self.model.model
        else:
            return self.model

    def get_wrapped_model(self):
        if self.is_peft:
            return self.model
        else:
            return self.model

    def get_model_config(self):
        if system.enable_deepspeed:
            return self.model.module.config
        else:
            return self.model.config

    def set_model_config(self, config):
        if system.enable_deepspeed:
            self.model.module.config = config
        else:
            self.model.config = config

    def before_pruning_step(self):
        pass

    def after_pruning_step(self):
        pass

    def during_pruning_step(self, module, mask, imp_metric, threshold):
        pass

    def before_pruning(self):
        pass

    def finishing_pruning(self):
        pass
