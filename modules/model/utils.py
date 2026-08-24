import os

import torch


def str_to_torch_dtype(dtype_str):
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "auto": 'auto',
        "float": torch.float,
        'None': None
    }
    return dtype_map.get(dtype_str, torch.float32)


def get_device_map():
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
    return device_map
