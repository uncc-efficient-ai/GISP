import json
import logging
from pathlib import Path

import numpy as np

import lm_eval

logger = logging.getLogger(__name__)


def lm_simple_eval(lm_eval_options, model, tokenizer, result_name, quick=False):
    if not isinstance(lm_eval_options.num_fewshot, list):
        lm_eval_options.num_fewshot = [lm_eval_options.num_fewshot]
    if quick and not lm_eval_options.quick_tasks:
        return
    for num_fewshot in lm_eval_options.num_fewshot:
        wrapped_model = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer,
                                                        batch_size=lm_eval_options.batch_size if lm_eval_options.batch_size and quick else 'auto')
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=wrapped_model,
            tasks=lm_eval_options.tasks if not quick else lm_eval_options.quick_tasks,
            # tasks=["openbookqa", "arc_easy", "winogrande", "hellaswag", "arc_challenge", "piqa", "boolq"],
            # tasks=["openbookqa"],
            num_fewshot=num_fewshot,
            log_samples=False,

        )

        def _handle_non_serializable(o):
            if isinstance(o, np.int64) or isinstance(o, np.int32):
                return int(o)
            elif isinstance(o, set):
                return list(o)
            else:
                return str(o)

        path = Path(lm_eval_options.output_path)
        # check if file or 'dir/results.json' exists
        if path.is_file():
            raise FileExistsError(f"File already exists at {path}")
        output_path_file = path.joinpath(f"{str(num_fewshot)}_{result_name}.json")
        if path.suffix in (".json", ".jsonl"):
            output_path_file = path
            path.parent.mkdir(parents=True, exist_ok=True)
            path = path.parent
        else:
            path.mkdir(parents=True, exist_ok=True)
        dumped = json.dumps(
            results, indent=2, default=_handle_non_serializable, ensure_ascii=False
        )
        output_path_file.open("w", encoding="utf-8").write(dumped)
        logger.info(
            f"lm_eval complete. See report in {str(num_fewshot)}_{result_name}.json located in {lm_eval_options.output_path}")
