import logging
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaConfig, \
    GPT2LMHeadModel, PhiForCausalLM, OPTForCausalLM, Qwen2Config, Qwen2ForCausalLM, MistralConfig, MistralForCausalLM

from .peft.make_peft_model import make_peft_model, import_customized_peft_module
from .utils import str_to_torch_dtype, get_device_map
from utils import load_module

logger = logging.getLogger(__name__)


def make_hf_model(model_config):
    if 'llama' in model_config.name or 'Llama' in model_config.name:
        if model_config.type == 'causal':
            if model_config.custom_modeling:
                config = LlamaConfig.from_pretrained(model_config.name, attn_implementation='eager')
                custom_package_module = load_module(model_config.custom_config.custom_package_location)
                model = custom_package_module.LlamaForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
                config = model.config
                logger.info(
                    f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
            else:
                # todo tokenizer.padding & spdaAttention in official modeling of llama
                config = LlamaConfig.from_pretrained(model_config.name, attn_implementation='eager')
                model = LlamaForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            # model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise NotImplementedError
    elif 'Mistral' in model_config.name or 'mistral' in model_config.name:
        if model_config.type == 'causal':
            if model_config.custom_modeling:
                config = MistralConfig.from_pretrained(model_config.name, attn_implementation='eager')
                custom_package_module = load_module(model_config.custom_config.custom_package_location)
                model = custom_package_module.MistralForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
                config = model.config
                logger.info(
                    f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
            else:
                # todo tokenizer.padding & spdaAttention in official modeling of llama
                config = MistralConfig.from_pretrained(model_config.name, attn_implementation='eager')
                model = MistralForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            # model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise NotImplementedError
    elif 'qwen' in model_config.name or 'Qwen' in model_config.name:
        if model_config.type == 'causal':
            if model_config.custom_modeling:
                config = Qwen2Config.from_pretrained(model_config.name, attn_implementation='eager')
                custom_package_module = load_module(model_config.custom_config.custom_package_location)
                model = custom_package_module.Qwen2ForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
                config = model.config
                logger.info(
                    f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
            else:
                # todo tokenizer.padding & spdaAttention in official modeling of llama
                config = Qwen2Config.from_pretrained(model_config.name, attn_implementation='eager')
                model = Qwen2ForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_config.name)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            # model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise NotImplementedError
    elif 'gpt2' in model_config.name:
        if model_config.type == 'causal':
            if model_config.custom_modeling:
                config = AutoConfig.from_pretrained(model_config.name, attn_implementation='eager')
                # config.resid_pdrop = 0
                # config.embd_pdrop = 0
                config.attn_pdrop = 0
                # config.activation_function = 'relu'
                custom_package_module = load_module(model_config.custom_config.custom_package_location)
                model = custom_package_module.GPT2LMHeadModel.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
                logger.info(
                    f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
            else:
                config = AutoConfig.from_pretrained(model_config.name, attn_implementation='eager')
                model = GPT2LMHeadModel.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_config.name, add_prefix_space=True)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            # model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise NotImplementedError
    elif 'phi-2' in model_config.name:
        if model_config.type == 'causal':
            if model_config.custom_modeling:
                config = AutoConfig.from_pretrained(model_config.name, attn_implementation='eager')
                custom_package_module = load_module(model_config.custom_config.custom_package_location)
                model = custom_package_module.PhiForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
                logger.info(
                    f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
            else:
                config = AutoConfig.from_pretrained(model_config.name, attn_implementation='eager')
                model = PhiForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_config.name, add_prefix_space=True)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            # model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise NotImplementedError
    elif 'opt' in model_config.name:
        if model_config.type == 'causal':
            if model_config.custom_modeling:
                config = AutoConfig.from_pretrained(model_config.name, attn_implementation='eager')
                custom_package_module = load_module(model_config.custom_config.custom_package_location)
                model = custom_package_module.OPTForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
                logger.info(
                    f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
            else:
                config = AutoConfig.from_pretrained(model_config.name, attn_implementation='eager')
                model = OPTForCausalLM.from_pretrained(
                    model_config.name,
                    config=config,
                    torch_dtype=str_to_torch_dtype(model_config.torch_dtype),
                    # device_map=get_device_map(),
                    load_in_8bit=model_config.load_in_8bit,
                )
            tokenizer = AutoTokenizer.from_pretrained(model_config.name, add_prefix_space=True)
            tokenizer.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tokenizer.padding_side = "left"  # Allow batched inference
            # model.config.pad_token_id = tokenizer.pad_token_id
        else:
            raise NotImplementedError

    if model_config.model_peft:
        model = make_peft_model(model_config, model)

    return model, tokenizer, config


def make_bin_model(model_config):
    model_task, tokenizer, config = make_hf_model(model_config)

    base_model = torch.load(model_config.model_path)
    tokenizer = base_model['tokenizer']
    # model = model['model']
    config = base_model['model'].config
    logger.info(
        f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
    logger.info(
        f"external customized {model_config.name} loaded from path: {model_config.model_path}")
    # model_task.base_model = base_model['model']
    model_task.model = base_model['model']
    del base_model
    # if model_config.custom_modeling:
    #
    # else:
    #     model = torch.load(model_config.model_path)
    #     tokenizer = model['tokenizer']
    #     # model = model['model']
    #     config = model.config
    #     model.base_model = model['model']
    torch.cuda.empty_cache()
    return model_task, tokenizer, config

# def make_ckpt_model(model_config):
#     model_task, tokenizer, config = make_hf_model(model_config)
#
#     if model_config.custom_modeling:
#         custom_package_module = load_module(model_config.custom_config.custom_package_location)
#         if model_config.model_peft:
#             import_customized_peft_module(model_config)
#         base_model = torch.load(model_config.model_path)
#         tokenizer = base_model['tokenizer']
#         # model = model['model']
#         config = base_model['model'].config
#         logger.info(
#             f"external customized {model_config.name} loaded: {model_config.custom_config.custom_package_location}")
#         logger.info(
#             f"external customized {model_config.name} loaded from path: {model_config.model_path}")
#         model_task.base_model = base_model['model']
#     else:
#         model = torch.load(model_config.model_path)
#         tokenizer = model['tokenizer']
#         # model = model['model']
#         config = model.config
#         model.base_model = model['model']
#     return model_task, tokenizer, config
