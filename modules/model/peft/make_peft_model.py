import importlib
import logging
import os
import peft

from utils import load_module

logger = logging.getLogger(__name__)


def make_peft_model(config, model):
    if config.peft.peft_type == 'lora':
        fan_in_fan_out = 'gpt2' in config.name
        # common settings
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType[config.peft.task_type.upper()],
            r=config.peft.lora.lora_r,
            lora_alpha=config.peft.lora.lora_alpha,
            lora_dropout=config.peft.lora.lora_dropout,
            target_modules=config.peft.lora.lora_target_modules,
            fan_in_fan_out=fan_in_fan_out
        )
        if config.peft.lora.custom_lora:
            if config.peft.lora.custom_config.custom_package_location != "":
                custom_package_module = load_module(config.peft.lora.custom_config.custom_package_location)
                logger.info(
                    f"external customized peft_config loaded: {config.peft.lora.custom_config.custom_package_location}")
                custom_peft_config = custom_package_module.apply_patch(config.peft, peft_config)
                peft_model = peft.get_peft_model(model, custom_peft_config)
            else:
                raise NotImplementedError
        else:
            logger.info(
                f"official peft_config loaded")
            peft_model = peft.get_peft_model(model, peft_config)
        # todo resume_from_checkpoint
        # if resume_from_checkpoint:
        #     # Check the available weights and load them
        #     checkpoint_name = os.path.join(
        #         resume_from_checkpoint, "pytorch_model.bin"
        #     )  # Full checkpoint
        #     if not os.path.exists(checkpoint_name):
        #         checkpoint_name = os.path.join(
        #             resume_from_checkpoint, "adapter_model.bin"
        #         )  # only LoRA model - LoRA config above has to fit
        #         resume_from_checkpoint = (
        #             False  # So the trainer won't try loading its state
        #         )
        #     # The two files above have a different name depending on how they were saved, but are actually the same.
        #     if os.path.exists(checkpoint_name):
        #         print(f"Restarting from {checkpoint_name}")
        #         adapters_weights = torch.load(checkpoint_name)
        #         model = set_peft_model_state_dict(model, adapters_weights)
        #     else:
        #         print(f"Checkpoint {checkpoint_name} not found")
        return peft_model
    else:
        raise NotImplementedError


def import_customized_peft_module(config,model):
    if config.peft.peft_type == 'lora':
        fan_in_fan_out = 'gpt2' in config.name
        # common settings
        peft_config = peft.LoraConfig(
            task_type=peft.TaskType[config.peft.task_type.upper()],
            r=config.peft.lora.lora_r,
            lora_alpha=config.peft.lora.lora_alpha,
            lora_dropout=config.peft.lora.lora_dropout,
            target_modules=config.peft.lora.lora_target_modules,
            fan_in_fan_out=fan_in_fan_out
        )
        if config.peft.lora.custom_lora:
            custom_package_module = load_module(config.peft.lora.custom_config.custom_package_location)
            logger.info(
                f"external customized peft_config loaded: {config.peft.lora.custom_config.custom_package_location}")
            custom_peft_config = custom_package_module.apply_patch(config.peft, peft_config)
            peft_model = peft.get_peft_model(model, custom_peft_config)
        else:
            raise NotImplementedError

        return peft_model
    else:
        raise NotImplementedError