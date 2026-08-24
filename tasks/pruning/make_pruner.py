import logging
import os

import transformers

from utils import load_module

logger = logging.getLogger(__name__)


def make_pruner(prune_config):
    if prune_config.custom_pruner:
        if prune_config.custom_config.custom_package_location != "":
            custom_package_module = load_module(prune_config.custom_config.custom_package_location)
            pruner_class = getattr(custom_package_module, prune_config.prune_metric)
            logger.info(
                f"external customized Pruner loaded: {prune_config.custom_config.custom_package_location}.{prune_config.prune_metric}")
            return pruner_class
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
