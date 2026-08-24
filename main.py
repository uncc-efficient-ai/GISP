import os
import random

import fire
import numpy as np
import psutil
import torch

import sys

import utils
from modules.config.config import Config

from pathlib import Path
import modules.system.system as system


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def main(config_path: str = ''):
    if not config_path:
        raise ValueError("config_path is required")

    config = Config(config_path)
    c = config.get_config()
    set_random_seed(c.task.seed)

    Path(c.task.output_folder).mkdir(parents=True, exist_ok=True)
    config.save_config(c.task.output_folder)
    system.init_system(c)

    # if c.report.use_logger:
    logger = utils.setup_logger(c.report, multi_process=system.world_size > 1)
    # logger = logging.getLogger(__name__)

    logger.info(f"loaded config: {config_path}")
    logger.info(c)
    logger.info(f"current cpu affinity:{psutil.cpu_count()}")

    # all_cpus = list(range(psutil.cpu_count()))
    # psutil.Process().cpu_affinity(all_cpus)
    if c.task.task_mode in ['fuse', 'tune']:
        from tasks.training.tune import tune_task
        tune_task(c)
    elif c.task.task_mode == 'prune':
        from tasks.pruning.prune import prune_task
        prune_task(c)
    elif c.task.task_mode == 'test':
        from tasks.test.test import test_task
        test_task(c)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    print(" ".join(sys.argv))
    fire.Fire(main)
