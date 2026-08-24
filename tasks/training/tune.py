
import logging
import os
from modules.data.make_data import make_train_data
from modules.eval.setup_eval import save_and_eval

from modules.model.make_model import make_model
from modules.reports.reports import setup_reports
import torch.distributed as dist

from tasks.training.make_trainer import make_trainer

logger = logging.getLogger(__name__)


def tune_task(c):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # setup report tools
    reporters = setup_reports(c.report)

    # load model
    model, tokenizer, config = make_model(c.model)

    train_data, eval_data = make_train_data(c.task.tune, c.model, c.task.seed, tokenizer)

    # customized trainer handle grad setup and all other stuff.
    trainer = make_trainer(c.task.tune.trainer, model, train_data, eval_data,
                           task_config=c, tokenizer=tokenizer)

    model.config.use_cache = False

    # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.train()

    print(
        "\n If there's a warning about missing keys above, please disregard :)"
    )

    if not ddp or dist.get_rank() == 0:
        save_and_eval(c, model, tokenizer, trainer)
