import logging
import os

import torch

logger = logging.getLogger(__name__)


def save(c, model, tokenizer):
    # torch.save({
    #     'model': model.base_model,
    #     'tokenizer': tokenizer
    # }, os.path.join(c.task.output_folder, "model.bin"))

    # for name, param in model.named_parameters():
    #     if not param.is_contiguous():
    #         print(f"Making {name} contiguous")
    #         param.data = param.data.contiguous()
    #
    # # save_pretrained might always be practical, but load_pretrained might face failure.
    # # for example, models with variant hidden_size or intermediate_size across layers. (couldn't init different shapes)
    # model.save_pretrained(c.task.output_folder)
    # tokenizer.save_pretrained(c.task.output_folder)

    logger.info(
        f"Model and tokenizer were saved as a Dict in {c.task.output_folder}/model.bin. \n You can load them like torch.load(\'{c.task.output_folder}/model.bin\')[\'model\']")


def eval(c, model, tokenizer):
    with torch.no_grad():
        model.eval()
        # Preserve the dtype selected by the model configuration.
        if c.evaluation.ppl:
            from .ppl import eval
            logger.info(eval(model, tokenizer))
            logger.info(f"ppl eval complete.")
        if c.evaluation.lm_eval:
            from .lm_eval_main import lm_simple_eval
            lm_simple_eval(c.evaluation.lm_eval_options, model, tokenizer, "lm_eval_result")
        if c.evaluation.commonsense_eval:
            from .commonsense_eval import main
            main(model, tokenizer, c.task.seed, c.evaluation.commonsense_eval_options.dataset,
                 c.evaluation.commonsense_eval_options.batch_size,
                 c.evaluation.commonsense_eval_options.output_path)
            logger.info(
                f"commonsense_eval precise pred complete. See report in acc.json located in {c.task.output_folder}")


def eval_ppl(model, tokenizer, save=False, save_path=None):
    from .ppl import eval
    result = eval(model, tokenizer)
    logger.info(result)
    logger.info(f"ppl eval complete.")
    if save:
        torch.save(result, save_path)
    return result


def eval_lm_eval(model, tokenizer, c, result_name, quick=False):
    from .lm_eval_main import lm_simple_eval
    lm_simple_eval(c.evaluation.lm_eval_options, model, tokenizer, result_name, quick=quick)



def save_and_eval(c, model, tokenizer, trainer=None):
    save(c, model, tokenizer)
    eval(c, model, tokenizer)
