import logging
import os

from modules.data.make_data import make_prune_data, make_train_data
from modules.eval.setup_eval import eval, save_and_eval

from modules.model.make_model import make_model
from modules.reports.reports import setup_reports
import modules.system.system as system

from tasks.pruning.make_pruner import make_pruner

logger = logging.getLogger(__name__)


def prune_task(c):
    # setup report tools
    reporters = setup_reports(c.report)

    # load model
    model, tokenizer, config = make_model(c.model)

    model.config.use_cache = False

    if c.task.prune.eval_before:
        logger.info(f"Running evaluation before pruning.")
        eval(c, model, tokenizer)

    prune_data = make_prune_data(c.task.prune, c.model, c.task.seed, tokenizer, model)

    pruner_c = make_pruner(c.task.prune)

    model = system.setup_model(model, prune_data, tokenizer, c)

    pruner = pruner_c(model, c, prune_data)

    pruner.tokenizer = tokenizer

    if c.task.tune:
        train_data, eval_data = make_train_data(c.task.tune, c.model, c.task.seed, tokenizer)
        pruner.train_data = train_data
        pruner.eval_data = eval_data

    pruner.prune()

    model = pruner.model

    if not system.ddp or system.rank == 0:
        save_and_eval(c, model, tokenizer)
