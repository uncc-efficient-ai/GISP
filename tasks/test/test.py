import logging
import os

from modules.eval.setup_eval import eval
import modules.system.system as system
from modules.model.make_model import make_model
from modules.reports.reports import setup_reports

logger = logging.getLogger(__name__)


def test_task(c):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # setup report tools
    reporters = setup_reports(c.report)

    # load model
    model, tokenizer, config = make_model(c.model)

    model = system.setup_model(model, None, tokenizer, c)

    eval(c, model, tokenizer)
