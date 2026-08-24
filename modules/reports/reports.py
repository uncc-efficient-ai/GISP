from . import wandb


def setup_reports(config):
    reps = {}
    if config.use_wandb:
        wandb.setup_env(config.wandb)
        reps['wandb'] = True
    if config.use_logger:
        reps['logger'] = True
    return reps
