import os


def setup_env(config):
    use_wandb = len(config.project) > 0 or (
            "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(config.project) > 0:
        os.environ["WANDB_PROJECT"] = config.project
    if len(config.watch) > 0:
        os.environ["WANDB_WATCH"] = config.watch
    if len(config.log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = config.log_model
    if len(config.run) > 0:
        os.environ["WANDB_NAME"] = config.run
    return use_wandb
