import logging
import os

import transformers

from utils import load_module

logger = logging.getLogger(__name__)


def make_trainer(trainer_config, model, train_data, eval_data, task_config, tokenizer):
    gradient_accumulation_steps = trainer_config.batch_size // trainer_config.micro_batch_size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    args = transformers.TrainingArguments(
        per_device_train_batch_size=trainer_config.micro_batch_size,
        per_device_eval_batch_size=trainer_config.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=trainer_config.warmup_steps,
        num_train_epochs=trainer_config.num_epochs,
        learning_rate=trainer_config.learning_rate,
        weight_decay=trainer_config.weight_decay,
        fp16=True,
        logging_steps=10,
        optim=trainer_config.optimizer,
        evaluation_strategy="steps" if len(eval_data) > 0 else "no",
        save_strategy="steps",
        eval_steps=trainer_config.eval_steps if len(eval_data) > 0 else None,
        save_steps=trainer_config.save_steps,
        output_dir=task_config.task.output_folder,
        save_total_limit=3,
        load_best_model_at_end=trainer_config.load_best_model_at_end,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=trainer_config.group_by_length,
        report_to="wandb" if task_config.report.use_wandb else None,
        run_name=task_config.report.wandb.run if task_config.report.use_wandb else None,
        seed=task_config.task.seed,
    )

    # todo more dataCollator such as CLM, RLM
    if trainer_config.collator == 'Seq2Seq':
        data_collator = transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        )
    elif trainer_config.collator == 'Data2Text':
        from modules.data.data_collator import DataCollatorForData2TextLanguageModeling
        data_collator = DataCollatorForData2TextLanguageModeling(
            tokenizer=tokenizer, mlm=False, mlm_probability=0.15,
            format_mode='cat'
        )
    else:
        raise NotImplementedError

    if trainer_config.custom_trainer:
        if trainer_config.custom_config.custom_trainer_location != "":
            custom_package_module = load_module(trainer_config.custom_config.custom_trainer_location)
            trainer = custom_package_module.MyTrainer(
                model=model,
                train_dataset=train_data,
                eval_dataset=eval_data,
                args=args,
                data_collator=data_collator,
                task_config=task_config, )
            logger.info(
                f"external customized Trainer loaded: {trainer_config.custom_config.custom_trainer_location}")
            return trainer
        else:
            raise NotImplementedError
    else:
        from transformers.trainer import Trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            args=args,
            data_collator=data_collator, )
        logger.info("default Trainer loaded")
        return trainer
