import os

import deepspeed
import torch
import psutil
world_size = 1
rank = 0
ddp = False
pipeline_nodes = [0]
device = 'cpu'
enable_pipeline = False
enable_deepspeed = False
deepspeed_config = None


def init_system(c):
    global world_size
    global rank
    global ddp
    global pipeline_nodes
    global device
    global enable_pipeline
    global enable_deepspeed
    global deepspeed_config

    device = c.system.device
    enable_pipeline = c.system.pipeline.enable_pipeline
    enable_deepspeed = c.system.deepspeed.enable_deepspeed

    if enable_pipeline:
        torch.distributed.init_process_group()
        world_size = int(os.environ.get("WORLD_SIZE", world_size))
        rank = int(os.environ.get("RANK", rank))
        ddp = enable_pipeline and world_size != 1
        pipeline_nodes = c.system.pipeline.pipeline_nodes
        device = torch.device(f"{device}:{rank % torch.cuda.device_count()}")
    elif enable_deepspeed:
        deepspeed.init_distributed()
        world_size = int(os.environ.get("WORLD_SIZE", world_size))
        rank = int(os.environ.get("RANK", rank))
        deepspeed_config = c.system.deepspeed.deepspeed_config


def setup_world(device, rank, world_size):
    return device


def setup_model(model, data, tokenizer, c):
    if enable_pipeline:
        model.to(device)
        from torch.distributed.pipelining import SplitPoint, pipeline, ScheduleGPipe

        layers_per_rank = model.config.num_hidden_layers // world_size
        print(f"layers_per_rank = {layers_per_rank}")
        split_spec = {
            f"model.layers.{i * layers_per_rank}": SplitPoint.BEGINNING
            for i in range(1, world_size)
        }
        temp_data = (torch.stack([d[0].view(-1) for d in data]))
        size_mbs = c.system.pipeline.size_mbs
        if size_mbs != 1:
            example_data = temp_data[0:size_mbs]
        else:
            example_data = temp_data[0].unsqueeze(0)
        pipe = pipeline(model, mb_args=(example_data,))
        stage = pipe.build_stage(rank, device=device)
        schedule = ScheduleGPipe(stage, c.task.prune.prune_dataset.n_samples // c.system.pipeline.num_mbs)
        return schedule
    elif enable_deepspeed:
        if c.task.task_mode in ['train']:
            model.train()
        model, optimizer, _, _ = deepspeed.initialize(
            model=model,
            config=deepspeed_config,
            model_parameters=model.parameters()
        )
    else:
        model.to(device)
    return model
