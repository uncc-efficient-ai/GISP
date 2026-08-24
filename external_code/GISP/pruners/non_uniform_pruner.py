import copy
import logging
import math
import sys
import traceback
from abc import abstractmethod

import deepspeed
from tqdm import tqdm
import torch

from .utils import *
from tasks.pruning.pruners import Pruner
from modules.eval.setup_eval import eval_ppl, eval_lm_eval

LAYER_NAME_MAPPING = {
    'Llama': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'Mistral': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'Qwen': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.o_proj',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'o_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'd': 'mlp.down_proj',
            'g': 'mlp.gate_proj',
            'u': 'mlp.up_proj',
            'd_name': 'down_proj',
            'g_name': 'gate_proj',
            'u_name': 'up_proj',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'phi-2': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.dense',
            'q_name': 'q_proj',
            'k_name': 'k_proj',
            'v_name': 'v_proj',
            'o_name': 'dense',
            'block': 'self_attn'
        },
        'mlp': {
            'u': 'mlp.fc1',
            'd': 'mlp.fc2',
            'u_name': 'fc1',
            'd_name': 'fc2',
            'block': 'mlp'
        },
        'layers': 'model.layers'
    },
    'opt': {
        'attn': {
            'q': 'self_attn.q_proj',
            'k': 'self_attn.k_proj',
            'v': 'self_attn.v_proj',
            'o': 'self_attn.out_proj',
            'block': 'self_attn'
        },
        'mlp': {
            'u': 'fc1',
            'd': 'fc2',
            'block': ''
        },
        'layers': 'base_model.decoder.layers'
    },
    # 'gpt2': {
    #     'o': 'attn.c_proj',
    #     'd': 'mlp.c_proj'
    # }

}


class non_uniform_pruner(Pruner):
    logger = logging.getLogger(__name__)

    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.label_neg = None
        self.data_neg = None
        self.label_pos = None
        self.data_pos = None
        self.pos = None
        self.label = None
        self.before_pruning_parameters = None
        self.use_cache = None
        for k, _ in LAYER_NAME_MAPPING.items():
            if k in config.model.name:
                self.layer_mapping = LAYER_NAME_MAPPING[k]
                self.model_arch = k
                self.model_name = config.model.alias
                break
        if self.layer_mapping is None:
            raise Exception(f'model {config.model.name} is not supported yet')
        self.W_metrics = {}
        self.tokenizer = None
        self.data_processed = False
        self.save_helper = InformationSaveHelper(config, logger)

    def min_max_scope_limit(self, r):
        if r < 0:
            if isinstance(r, torch.Tensor):
                r[()] = 0
            else:
                r = 0
        if r >= self.config.task.prune.max_ratio:
            # dif = r - self.config.task.prune.max_ratio
            if isinstance(r, torch.Tensor):
                r[()] = self.config.task.prune.max_ratio
            else:
                r = self.config.task.prune.max_ratio
        return r

    def obtain_information(self, input=None):
        n_samples = self.config.task.prune.prune_dataset.n_samples
        seq_len = self.config.task.prune.prune_dataset.seq_len
        self.model.eval()
        self.model.config.use_cache = False

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(self.model, self.data, n_samples,
                                                                                 seq_len)
        if input is not None:
            inps = input

        def forward_layer(layer, inputs):

            with torch.no_grad():
                if isinstance(layer, nn.Identity):
                    outputs = layer(inputs)
                else:
                    outputs = inputs.detach().clone()
                    for j in range(n_samples):
                        outputs[j] = \
                            layer(inputs[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[
                                0]
                    # outputs = \
                    #     layer(inputs, attention_mask=attention_mask, position_ids=position_ids)[
                    #         0]
            return outputs

        layers = nested_getattr(self.model, self.layer_mapping['layers'])
        current_cos_sim = []
        current_l2_dis = []
        current_l2_dis_token = []
        current_angular_norm = []
        current_std_dis = []
        current_std_dis_token = []
        current_mean_dis = []
        current_mean_dis_token = []
        current_cross_product = []
        current_kl = []
        inputs = inps.detach().clone()
        for j in tqdm(range(0, len(layers)), desc="Obtaining following layers' cosine similarity"):
            current_layer = layers[j]
            outputs = forward_layer(current_layer, inputs)
            current_cos_sim.append(cosine_similarity(inputs, outputs)[2])
            l2, l2_t = l2_distance(inputs.float(), outputs.float())
            current_l2_dis.append(l2)
            current_l2_dis_token.append(l2_t)

            std_v, std_t = std(inputs.float(), outputs.float())
            current_std_dis.append(std_v)
            current_std_dis_token.append(std_t)

            m, m_t = mean(inputs.float(), outputs.float())
            current_mean_dis.append(m)
            current_mean_dis_token.append(m_t)

            # kl = kl_divergence(inputs, outputs)
            # current_kl.append(kl)
            current_cross_product.append(dot_product_similarity(inputs.float(), outputs.float())[2])

            current_angular_norm.append(angular_distance(inputs, outputs))
            inputs, outputs = outputs, inputs
        current_angular_norm = torch.tensor(current_angular_norm)
        current_cos_sim = torch.tensor(current_cos_sim)
        current_l2_dis = torch.tensor(current_l2_dis)
        current_std_dis = torch.tensor(current_std_dis)
        # current_kl = torch.tensor(current_kl)
        show(current_cos_sim, 'cosine_sim', f'cosine_sim in {self.model_name}')
        show(current_l2_dis, 'l2_distance', f'l2_distance in {self.model_name}')
        show(current_std_dis, 'abs( std(in) - std(out) )', f'std change in {self.model_name}')
        show(current_angular_norm, 'angular_dis', f'angular_dis in {self.model_name}')
        show(current_cross_product, 'cross product', f'cross product in {self.model_name}')
        # show(current_kl, 'kl divergence', f'angular_dis in {self.model_name}')
        return current_cos_sim, current_l2_dis, current_angular_norm, current_std_dis

    def get_layers(self):
        return nested_getattr(self.get_model(), self.layer_mapping['layers'])

    def fill_information(self, cpu_grads=False):
        for param in self.get_wrapped_model().parameters():
            param.requires_grad_(True)
        self.get_wrapped_model().zero_grad()

        def buffer_grad_norm():
            for module_param in self.get_wrapped_model().parameters():
                if hasattr(module_param, 'grad_norm'):
                    if self.config.task.prune.grad_norm_type in ['l1']:
                        module_param.grad_norm += torch.abs(module_param.grad)
                    elif self.config.task.prune.grad_norm_type in ['l2']:
                        module_param.grad_norm += module_param.grad ** 2
                else:
                    if self.config.task.prune.grad_norm_type in ['l1']:
                        module_param.grad_norm = copy.deepcopy(torch.abs(module_param.grad))
                    elif self.config.task.prune.grad_norm_type in ['l2']:
                        module_param.grad_norm = copy.deepcopy(module_param.grad ** 2)

        def replace_grad_norm():
            for module_param in self.get_wrapped_model().parameters():
                if hasattr(module_param, 'grad_norm'):
                    if self.config.task.prune.grad_norm_type in ['l1']:
                        module_param.grad = module_param.grad_norm
                    elif self.config.task.prune.grad_norm_type in ['l2']:
                        module_param.grad = torch.sqrt(module_param.grad_norm)
                    if self.config.task.prune.scale_grad_norm:
                        module_param.grad = module_param.grad / n_samples
                    del module_param.grad_norm

        if not self.data_processed:
            if self.config.task.prune.prune_dataset.type in ['downstream']:
                multiple_lengths = len({inp.size(1) for inp, _, _ in self.data}) > 1
                device = self.get_wrapped_model().device
                if multiple_lengths:
                    from collections import defaultdict
                    buckets_pos, buckets_neg = defaultdict(list), defaultdict(list)
                    for inp, lbl, pos in self.data:
                        L = inp.size(1)
                        if pos:
                            buckets_pos[L].append((inp, lbl))
                        else:
                            buckets_neg[L].append((inp, lbl))
                    self.data_pos, self.label_pos = {}, {}
                    self.data_neg, self.label_neg = {}, {}
                    for L, pairs in buckets_pos.items():
                        big_inp = torch.cat([i for i, _ in pairs], dim=0).to(device)
                        big_lbl = torch.cat([l for _, l in pairs], dim=0).to(device)
                        self.data_pos[L] = big_inp
                        self.label_pos[L] = big_lbl
                    for L, pairs in buckets_neg.items():
                        big_inp = torch.cat([i for i, _ in pairs], dim=0).to(device)
                        big_lbl = torch.cat([l for _, l in pairs], dim=0).to(device)
                        self.data_neg[L] = big_inp
                        self.label_neg[L] = big_lbl
                else:
                    # 全部长度相同，直接 stack 再分正负
                    all_inp = torch.cat([i for i, _, _ in self.data], dim=0).to(device)  # [N, L]
                    all_lbl = torch.cat([l for _, l, _ in self.data], dim=0).to(device)  # [N, L]
                    all_pos = [pos for _, _, pos in self.data]

                    # 选出正例、反例索引
                    idxs_pos = [i for i, flag in enumerate(all_pos) if flag]
                    idxs_neg = [i for i, flag in enumerate(all_pos) if not flag]

                    L = all_inp.size(1)
                    # 只保留正例
                    if idxs_pos:
                        self.data_pos = {L: all_inp[idxs_pos]}
                        self.label_pos = {L: all_lbl[idxs_pos]}
                    else:
                        self.data_pos, self.label_pos = {}, {}
                    # 只保留反例
                    if idxs_neg:
                        self.data_neg = {L: all_inp[idxs_neg]}
                        self.label_neg = {L: all_lbl[idxs_neg]}
                    else:
                        self.data_neg, self.label_neg = {}, {}
                self.data_processed = True
            else:
                self.data = torch.stack([data[0].view(-1) for data in self.data])
                self.data = self.data.to(self.get_wrapped_model().device)
                self.data_processed = True

        avg_loss = []
        batch_size = self.config.task.prune.batch_size
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {available_gpus}")
        base_model = self.get_wrapped_model()
        if available_gpus > 2:
            # available_gpus = 8
            if self.config.task.prune.prune_dataset.type in ['downstream']:
                devices = [f"cuda:{i}" for i in range(min(available_gpus, torch.cuda.device_count()))]
                world_size = len(devices)
                models = []

                for i, dev in enumerate(devices):
                    if i == 0:
                        model_i = base_model
                    else:
                        model_i = copy.deepcopy(base_model)
                    model_i.to(dev)
                    models.append(model_i)

                buckets_pos = [[] for _ in range(world_size)]
                buckets_neg = [[] for _ in range(world_size)]

                for L, inps in self.data_pos.items():
                    labs = self.label_pos[L]  # [N, L]
                    in_splits = torch.tensor_split(inps, world_size, dim=0)
                    lab_splits = torch.tensor_split(labs, world_size, dim=0)
                    for k in range(world_size):
                        if in_splits[k].numel() == 0:  # 该卡该 L 空
                            continue
                        buckets_pos[k].append((L,
                                               in_splits[k].to(devices[k], non_blocking=True),
                                               lab_splits[k].to(devices[k], non_blocking=True)))

                for L, inps in self.data_neg.items():
                    labs = self.label_neg[L]
                    in_splits = torch.tensor_split(inps, world_size, dim=0)
                    lab_splits = torch.tensor_split(labs, world_size, dim=0)
                    for k in range(world_size):
                        if in_splits[k].numel() == 0:
                            continue
                        buckets_neg[k].append((L,
                                               in_splits[k].to(devices[k], non_blocking=True),
                                               lab_splits[k].to(devices[k], non_blocking=True)))

                def process_on_device(buckets, model, device, pos):
                    # model.to(device)
                    # loss_sum = torch.zeros((), device=device)
                    losses = []
                    # for L, inps, labs in tqdm(buckets, desc=f"Buckets on {device}"):
                    for L, inps, labs in buckets:
                        N = inps.size(0)
                        if L > 512:
                            batch_size = 4
                        elif L > 256:
                            batch_size = 4
                        elif 128 < L <= 256:
                            batch_size = 8
                        elif 64 < L <= 128:
                            batch_size = 16
                        elif 32 <= L <= 64:
                            batch_size = 32
                        else:
                            batch_size = 64
                        # for j in tqdm(range(0, N, batch_size), desc=f"{device} L={L}"):
                        for j in range(0, N, batch_size):
                            end = min(j + batch_size, N)
                            batch_in = inps[j:end]
                            batch_lab = labs[j:end]
                            if batch_in.dim() == 1:
                                batch_in = batch_in.unsqueeze(0)
                                batch_lab = batch_lab.unsqueeze(0)
                            loss = model(batch_in, labels=batch_lab).loss * batch_in.size(0)
                            loss.backward()
                            losses.append(loss.detach())
                            # logger.info(f"{device} {pos} Loss = {loss.item()}")
                    losses = [l.item() for l in losses]
                    return losses

                # 3) NEG 阶段：清 grad，起 N 个线程
                for m in models:
                    for p in m.parameters():
                        if hasattr(p, 'grad_neg'):
                            p.grad_neg = None
                        p.grad = None

                results_neg = [None] * world_size
                import threading

                def handle_thread_exception(args):
                    tb_str = ''.join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
                    logger.info(f"\nThread {args.thread.name} crashed with exception:\n{args.exc_value}\n{tb_str}")
                    sys.exit(1)  # 直接退出整个程序

                threading.excepthook = handle_thread_exception

                threads = []
                for k in range(world_size):
                    t = threading.Thread(target=lambda idx=k: results_neg.__setitem__(
                        idx, process_on_device(buckets_neg[idx], models[idx], devices[idx], False)))
                    threads.append(t)
                    t.start()
                for t in threads: t.join()

                # 4) 把各卡 neg 的 grad 汇总到 0 卡，再存到 p0.grad_neg（与原语义一致：放 CPU）
                with torch.cuda.device(devices[0]):
                    for params in zip(*(m.parameters() for m in models)):
                        grads = []
                        for p in params:
                            if p.grad is not None:
                                grads.append(p.grad.to(devices[0]))
                            p.grad = None  # 顺手清掉
                        if grads:
                            acc = grads[0]
                            for g in grads[1:]:
                                acc = acc + g
                            params[0].grad_neg = acc.detach().cpu()
                        else:
                            params[0].grad_neg = None

                avg_loss_neg = torch.mean(torch.tensor([x for lst in results_neg for x in (lst or [])]))

                # 5) POS 阶段：保留 grad_neg，清 grad，起 N 个线程
                for m in models:
                    for p in m.parameters():
                        p.grad = None

                results_pos = [None] * world_size
                threads = []
                for k in range(world_size):
                    t = threading.Thread(target=lambda idx=k: results_pos.__setitem__(
                        idx, process_on_device(buckets_pos[idx], models[idx], devices[idx], True)))
                    threads.append(t)
                    t.start()
                for t in threads: t.join()

                # 6) 把各卡 pos 的 grad 累到 0 卡的 p0.grad（和你现在两卡累加一致）
                with torch.cuda.device(devices[0]):
                    for params in zip(*(m.parameters() for m in models)):
                        acc = None
                        for p in params:
                            if p.grad is not None:
                                if acc is None:
                                    acc = p.grad.to(devices[0])
                                else:
                                    acc.add_(p.grad.to(devices[0]))
                            p.grad = None
                        params[0].grad = acc  # 允许为 None

                avg_loss = torch.mean(torch.tensor([x for lst in results_pos for x in (lst or [])]))

                # 7) 收尾
                self.model = models[0]
                for m in models[1:]:
                    del m
            else:
                devices = [f"cuda:{i}" for i in range(min(available_gpus, torch.cuda.device_count()))]
                world_size = len(devices)

                # 模型副本
                models = []
                for i, dev in enumerate(devices):
                    if i == 0:
                        m = base_model
                    else:
                        m = copy.deepcopy(base_model)
                    m.to(dev)
                    models.append(m)

                # 把 self.data 按 batch 维均分到各卡
                splits = torch.tensor_split(self.data, world_size, dim=0)
                data_dev = [splits[i].to(devices[i], non_blocking=True) for i in range(world_size)]

                def process_on_gpuX(data_tensor, model, device):
                    avg_loss_gpu = []
                    N = len(data_tensor)
                    L = data_tensor[0].shape[0]
                    if L > 512:
                        batch_size = 1
                    elif L > 256:
                        batch_size = 4
                    elif 128 < L <= 256:
                        batch_size = 8
                    elif 64 < L <= 128:
                        batch_size = 16
                    elif 32 <= L <= 64:
                        batch_size = 32
                    else:
                        batch_size = 64
                    # for j in tqdm(range(0, N, batch_size), desc=f"Processing on {device}"):
                    for j in range(0, N, batch_size):
                        end_idx = min(j + batch_size, N)
                        batch_input = data_tensor[j:end_idx]
                        loss = model(batch_input, labels=batch_input).loss * batch_input.size(0)
                        loss.backward()
                        avg_loss_gpu.append(loss)
                        # logger.info(f"{device} Loss = {loss.item()}")
                    avg_loss_gpu = [l.item() for l in avg_loss_gpu]
                    return avg_loss_gpu

                # 清 grad
                for m in models:
                    for p in m.parameters():
                        p.grad = None

                results = [None] * world_size
                import threading
                threads = []
                for k in range(world_size):
                    t = threading.Thread(target=lambda idx=k: results.__setitem__(
                        idx, process_on_gpuX(data_dev[idx], models[idx], devices[idx])))
                    threads.append(t);
                    t.start()
                for t in threads: t.join()

                # 累加各卡 grad 到 0 卡
                with torch.cuda.device(devices[0]):
                    for params in zip(*(m.parameters() for m in models)):
                        acc = None
                        for p in params:
                            if p.grad is not None:
                                if acc is None:
                                    acc = p.grad.to(devices[0])
                                else:
                                    acc.add_(p.grad.to(devices[0]))
                            p.grad = None
                        params[0].grad = acc

                avg_loss = torch.mean(torch.tensor([x for lst in results for x in (lst or [])]))

                self.model = models[0]
                for m in models[1:]:
                    del m
                torch.cuda.empty_cache()
        elif available_gpus == 2:
            if self.config.task.prune.prune_dataset.type in ['downstream']:
                model_gpu0 = base_model
                model_gpu1 = copy.deepcopy(base_model)

                # 先分别为正例和负例准备 0/1 号 GPU 的 bucket
                buckets0_pos, buckets1_pos = [], []
                buckets0_neg, buckets1_neg = [], []

                # 正例部分
                for L, inps in self.data_pos.items():
                    labs = self.label_pos[L]  # [N, L]
                    N = inps.size(0)
                    h = N // 2

                    if h <= 0:
                        # 全部放 GPU0
                        buckets0_pos.append((L,
                                             inps.to("cuda:0", non_blocking=True),
                                             labs.to("cuda:0", non_blocking=True)))
                    else:
                        # 前半放 GPU0，后半放 GPU1
                        buckets0_pos.append((L,
                                             inps[:h].to("cuda:0", non_blocking=True),
                                             labs[:h].to("cuda:0", non_blocking=True)))
                        buckets1_pos.append((L,
                                             inps[h:].to("cuda:1", non_blocking=True),
                                             labs[h:].to("cuda:1", non_blocking=True)))

                # 负例部分
                for L, inps in self.data_neg.items():
                    labs = self.label_neg[L]
                    N = inps.size(0)
                    h = N // 2

                    if h <= 0:
                        buckets0_neg.append((L,
                                             inps.to("cuda:0", non_blocking=True),
                                             labs.to("cuda:0", non_blocking=True)))
                    else:
                        buckets0_neg.append((L,
                                             inps[:h].to("cuda:0", non_blocking=True),
                                             labs[:h].to("cuda:0", non_blocking=True)))
                        buckets1_neg.append((L,
                                             inps[h:].to("cuda:1", non_blocking=True),
                                             labs[h:].to("cuda:1", non_blocking=True)))

                def process_on_device(buckets, model, device, pos):
                    model.to(device)
                    losses = []
                    for L, inps, labs in tqdm(buckets, desc=f"Buckets on {device}"):
                        N = inps.size(0)
                        if L > 512:
                            batch_size = 1
                        elif L > 256:
                            batch_size = 4
                        elif 128 < L <= 256:
                            batch_size = 8
                        elif 64 < L <= 128:
                            batch_size = 16
                        elif 32 <= L <= 64:
                            batch_size = 32
                        else:
                            batch_size = 64
                        for j in tqdm(range(0, N, batch_size),
                                      desc=f"{device} L={L}"):
                            end = min(j + batch_size, N)
                            batch_in = inps[j:end]  # 已在对应 GPU
                            batch_lab = labs[j:end]
                            if batch_in.dim() == 1:
                                batch_in = batch_in.unsqueeze(0)
                                batch_lab = batch_lab.unsqueeze(0)
                            loss = model(batch_in, labels=batch_lab).loss * batch_in.size(0)
                            loss.backward()
                            losses.append(loss.item())
                            logger.info(f"{device} {pos} Loss = {loss.item()}")
                    return losses

                results = [None, None, None, None]

                import threading
                def worker_on_gpu0():
                    results[0] = process_on_device(buckets0_pos, model_gpu0, "cuda:0", True)

                def worker_on_gpu1():
                    results[1] = process_on_device(buckets1_pos, model_gpu1, "cuda:1", True)

                def worker_on_gpu0_neg():
                    results[2] = process_on_device(buckets0_neg, model_gpu0, "cuda:0", False)

                def worker_on_gpu1_neg():
                    results[3] = process_on_device(buckets1_neg, model_gpu1, "cuda:1", False)

                for p in model_gpu0.parameters():
                    if hasattr(p, 'grad_neg'):
                        p.grad_neg = None
                    p.grad = None

                for p in model_gpu1.parameters():
                    if hasattr(p, 'grad_neg'):
                        p.grad_neg = None
                    p.grad = None

                t0 = threading.Thread(target=worker_on_gpu0)
                t1 = threading.Thread(target=worker_on_gpu1)

                t3 = threading.Thread(target=worker_on_gpu0_neg)
                t4 = threading.Thread(target=worker_on_gpu1_neg)

                t3.start()
                t4.start()
                t3.join()
                t4.join()
                avg_loss_neg = results[2] + results[3]
                avg_loss_neg = torch.mean(torch.tensor(avg_loss_neg))

                for p0, p1 in zip(model_gpu0.parameters(), model_gpu1.parameters()):
                    g0 = p0.grad if p0.grad is not None else torch.zeros_like(p0)
                    g1 = p1.grad.to("cuda:0") if p1.grad is not None else torch.zeros_like(p0)
                    p0.grad_neg = (g0 + g1).cpu()
                    p0.grad = None
                    p1.grad = None

                t0.start()
                t1.start()
                t0.join()
                t1.join()
                avg_loss = results[0] + results[1]
                avg_loss = torch.mean(torch.tensor(avg_loss))
                # 把 GPU1 上跑正样本的梯度搬回 GPU0，累加到 grad
                for p0, p1 in zip(model_gpu0.parameters(), model_gpu1.parameters()):
                    if p1.grad is not None:
                        if p0.grad is None:
                            p0.grad = p1.grad.to("cuda:0")
                        else:
                            p0.grad += p1.grad.to("cuda:0")
                    if hasattr(p0, 'grad_neg'):
                        p0.grad_neg = p0.grad_neg.to("cuda:0")
                del model_gpu1
                self.model = model_gpu0
            else:
                model_gpu0 = base_model
                model_gpu1 = copy.deepcopy(base_model).to("cuda:1")
                half_point = len(self.data) // 2
                data_gpu0 = self.data[:half_point]
                data_gpu1 = self.data[half_point:].to("cuda:1")

                def process_on_gpu0():
                    avg_loss_gpu0 = []
                    for j in tqdm(range(0, len(data_gpu0), batch_size), desc="Processing on GPU 0"):
                        end_idx = min(j + batch_size, len(data_gpu0))
                        batch_input = data_gpu0[j:end_idx]
                        loss = model_gpu0(batch_input, labels=batch_input).loss * batch_input.size()[0]
                        loss.backward()
                        avg_loss_gpu0.append(loss.item())
                        logger.info(f"GPU 0 Loss = {loss.item()}")
                    return avg_loss_gpu0

                def process_on_gpu1():
                    avg_loss_gpu1 = []
                    for j in tqdm(range(0, len(data_gpu1), batch_size), desc="Processing on GPU 1"):
                        end_idx = min(j + batch_size, len(data_gpu1))
                        batch_input = data_gpu1[j:end_idx]
                        loss = model_gpu1(batch_input, labels=batch_input).loss * batch_input.size()[0]
                        loss.backward()
                        avg_loss_gpu1.append(loss.item())
                        logger.info(f"GPU 1 Loss = {loss.item()}")
                    return avg_loss_gpu1

                results = [None, None]

                def thread_gpu0():
                    results[0] = process_on_gpu0()

                def thread_gpu1():
                    results[1] = process_on_gpu1()

                import threading
                t0 = threading.Thread(target=thread_gpu0)
                t1 = threading.Thread(target=thread_gpu1)

                t0.start()
                t1.start()

                # 等待两个线程完成
                t0.join()
                t1.join()
                avg_loss = results[0] + results[1]

                for param0, param1 in zip(model_gpu0.parameters(), model_gpu1.parameters()):
                    if param0.grad is not None and param1.grad is not None:
                        param0.grad = param0.grad + param1.grad.to("cuda:0")
                del model_gpu1
                self.model = model_gpu0
        else:
            if self.config.task.prune.prune_dataset.type in ['downstream']:
                device = base_model.device
                buckets0_pos = []
                buckets0_neg = []

                # 正例部分
                for L, inps in self.data_pos.items():
                    labs = self.label_pos[L]  # [N, L]
                    # 全部放 GPU0
                    buckets0_pos.append((L,
                                         inps.to(device, non_blocking=True),
                                         labs.to(device, non_blocking=True)))
                # 负例部分
                for L, inps in self.data_neg.items():
                    labs = self.label_neg[L]
                    buckets0_neg.append((L,
                                         inps.to(device, non_blocking=True),
                                         labs.to(device, non_blocking=True)))

                for p in base_model.parameters():
                    if hasattr(p, 'grad_neg'):
                        p.grad_neg = None
                    p.grad = None

                avg_loss_neg = []

                for L, inps, labs in tqdm(buckets0_neg, desc=f"Buckets on {device}"):
                    N = inps.size(0)
                    if L > 512:
                        batch_size = 1
                    elif L > 256:
                        batch_size = 4
                    elif 128 < L <= 256:
                        batch_size = 8
                    elif 64 < L <= 128:
                        batch_size = 16
                    elif 32 <= L <= 64:
                        batch_size = 32
                    else:
                        batch_size = 64
                    for j in tqdm(range(0, N, batch_size),
                                  desc=f"{device} L={L}"):
                        end = min(j + batch_size, N)
                        batch_in = inps[j:end]  # 已在对应 GPU
                        batch_lab = labs[j:end]
                        if batch_in.dim() == 1:
                            batch_in = batch_in.unsqueeze(0)
                            batch_lab = batch_lab.unsqueeze(0)
                        loss = base_model(batch_in, labels=batch_lab).loss * batch_in.size(0)
                        avg_loss_neg.append(loss.item())
                        loss.backward()
                        logger.info(f"{device} {False} Loss = {loss.item()}")

                for p0 in base_model.parameters():
                    g0 = p0.grad if p0.grad is not None else torch.zeros_like(p0)
                    p0.grad_neg = g0.cpu()
                    p0.grad = None

                for L, inps, labs in tqdm(buckets0_pos, desc=f"Buckets on {device}"):
                    N = inps.size(0)
                    if L > 512:
                        batch_size = 1
                    elif L > 256:
                        batch_size = 4
                    elif 128 < L <= 256:
                        batch_size = 8
                    elif 64 < L <= 128:
                        batch_size = 16
                    elif 32 <= L <= 64:
                        batch_size = 32
                    else:
                        batch_size = 64
                    for j in tqdm(range(0, N, batch_size),
                                  desc=f"{device} L={L}"):
                        end = min(j + batch_size, N)
                        batch_in = inps[j:end]  # 已在对应 GPU
                        batch_lab = labs[j:end]
                        if batch_in.dim() == 1:
                            batch_in = batch_in.unsqueeze(0)
                            batch_lab = batch_lab.unsqueeze(0)
                        loss = base_model(batch_in, labels=batch_lab).loss * batch_in.size(0)
                        avg_loss.append(loss.item())
                        loss.backward()
                        logger.info(f"{device} {True} Loss = {loss.item()}")
            else:
                n_samples = self.config.task.prune.prune_dataset.n_samples
                if batch_size == 1:
                    for j in tqdm(range(n_samples), desc="Obtaining first-order grad information."):
                        batch_input = self.data[j].unsqueeze(0)
                        loss = self.get_wrapped_model()(batch_input, labels=batch_input).loss
                        avg_loss.append(loss.item())
                        logger.info("Loss = {}".format(loss))
                        loss.backward()
                        if self.config.task.prune.grad_norm_sample:
                            buffer_grad_norm()
                else:
                    for j in tqdm(range(0, n_samples, batch_size), desc="Obtaining first-order grad information."):
                        end_idx = min(j + batch_size, n_samples)
                        batch_input = self.data[j:end_idx]
                        actual_batch_size = len(batch_input)
                        loss = self.get_wrapped_model()(batch_input, labels=batch_input).loss * actual_batch_size
                        avg_loss.append(loss.item())
                        logger.info("Loss = {}".format(loss))
                        loss.backward()
                        if self.config.task.prune.grad_norm_sample:
                            buffer_grad_norm()
                if self.config.task.prune.grad_norm_sample:
                    replace_grad_norm()
            avg_loss = torch.mean(torch.tensor(avg_loss))
        return avg_loss

    def ratio_scheduling(self, iteration):

        pruning_ratio = self.config.task.prune.ratio
        if iteration == 1:
            ratios = torch.tensor([pruning_ratio], device=self.get_wrapped_model().device, dtype=torch.float32)
        else:
            if self.config.task.prune.iterative_scheduling in ['linear']:
                ratios = torch.linspace(0 if iteration != 1 else pruning_ratio, pruning_ratio, iteration,
                                        device=self.get_wrapped_model().device)
            elif self.config.task.prune.iterative_scheduling in ['cosine']:
                steps = torch.arange(iteration, device=self.get_wrapped_model().device, dtype=torch.float32)
                ratios = pruning_ratio * 0.5 * (1 - torch.cos(math.pi * steps / (iteration - 1)))
            elif self.config.task.prune.iterative_scheduling in ['exp']:
                alpha = self.config.task.prune.exp_alpha
                steps = torch.arange(iteration, device=self.get_wrapped_model().device, dtype=torch.float32)
                if self.config.task.prune.exp_shape == 'concave':
                    # 先快后慢
                    exp_values = 1 - torch.exp(-alpha * steps)
                    max_exp = 1 - math.exp(-alpha * iteration)
                    ratios = pruning_ratio * (exp_values / max_exp)
                elif self.config.task.prune.exp_shape == 'convex':
                    # 先慢后快
                    ratios = pruning_ratio * (
                            (torch.exp(alpha * (steps - (iteration - 1))) - math.exp(-alpha * (iteration - 1)))
                            / (1 - math.exp(-alpha * (iteration - 1))))
            elif self.config.task.prune.iterative_scheduling in ['lottery_ticket_p']:
                alpha = self.config.task.prune.exp_alpha
                steps = torch.arange(iteration, device=self.get_wrapped_model().device, dtype=torch.float32)
                if self.config.task.prune.exp_shape == 'concave':
                    # 先快后慢
                    exp_values = 1 - torch.exp(-alpha * steps)
                    max_exp = 1 - math.exp(-alpha * iteration)
                    ratios = pruning_ratio * (exp_values / max_exp)
                elif self.config.task.prune.exp_shape == 'convex':
                    # 先慢后快
                    ratios = pruning_ratio * (
                            (torch.exp(alpha * (steps - (iteration - 1))) - math.exp(-alpha * (iteration - 1)))
                            / (1 - math.exp(-alpha * (iteration - 1))))
            else:
                raise NotImplementedError
        return ratios

    def substract_mlp(self, layer):
        return getattr(layer, self.layer_mapping['mlp']['block'])

    def substract_attn(self, layer):
        return getattr(layer, self.layer_mapping['attn']['block'])

    @abstractmethod
    def prune(self):
        pass

    @abstractmethod
    def step(self):
        pass

    @abstractmethod
    def get_imps(self):
        pass

    def evaluation(self, current_ratio, title=None, quick=False):
        eval_ppl(self.get_wrapped_model(), self.tokenizer, save=True,
                 save_path=title if title is not None else f'{self.config.task.output_folder}/sp_{current_ratio}_ppl.pth')
        if quick:
            eval_lm_eval(self.get_wrapped_model(), self.tokenizer, self.config,
                         title if title is not None else f'sp_{current_ratio}_lm_eval_quick_0', quick=True)
        else:
            eval_lm_eval(self.get_wrapped_model(), self.tokenizer, self.config,
                         title if title is not None else f'sp_{current_ratio}_lm_eval', quick=False)

    def before_pruning(self):
        self.get_wrapped_model().eval()
        self.use_cache = self.get_model_config().use_cache
        self.get_model_config().use_cache = False
        self.before_pruning_parameters = self.count_params()

    def check_sparsity(self, real_pruning=True):
        if real_pruning:
            after_pruning_params = self.count_params()
            logger.info(f"{100 - 100.0 * after_pruning_params / self.before_pruning_parameters}%")
        else:
            after_pruning_params = self.check_unstr_sparsity()[2]
            logger.info(f"{100.0 * after_pruning_params / self.before_pruning_parameters}%")
        return 1 - (after_pruning_params / self.before_pruning_parameters)

    def after_pruning_step(self, current_ratio=None, real_pruning=True):
        layers = self.get_layers()
        r = self.check_sparsity(real_pruning)
        # self.evaluation(r, quick=True)
        return r

    def finishing_pruning(self, real_pruning=True):
        self.get_model_config().use_cache = self.use_cache
        torch.cuda.empty_cache()
        r = self.check_sparsity(real_pruning)
        self.evaluation(r, quick=False)

    def check_unstr_sparsity(self):
        logger.info("*" * 30)
        count = 0
        total_params = 0
        layers = self.get_layers()
        block = self.config.task.prune.prune_modules
        for i in range(len(layers)):
            layer = layers[i]

            if block in ['mlp']:
                mlp_block = self.substract_mlp(layer)
                subset = find_layers(mlp_block)
            elif block in ['mha']:
                attn_block = self.substract_attn(layer)
                subset = find_layers(attn_block)
            else:
                subset = find_layers(layer)
            sub_count = 0
            sub_params = 0
            for name in subset:
                W = subset[name].weight.data
                count += (W == 0).sum().item()
                total_params += W.numel()

                sub_count += (W == 0).sum().item()
                sub_params += W.numel()

            logger.info(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
        logger.info("*" * 30)
        return float(count) / total_params, count, total_params - count

    def count_params(self):
        layer_params = 0
        layers = self.get_layers()
        block = self.config.task.prune.prune_modules
        if block in ['mlp']:
            for l in layers:
                mlp_block = self.substract_mlp(l)
                layer_params += sum(p.numel() for p in mlp_block.parameters())
        elif block in ['mha']:
            for l in layers:
                attn_block = self.substract_attn(l)
                layer_params += sum(p.numel() for p in attn_block.parameters())
        else:
            for l in layers:
                layer_params += sum(p.numel() for p in l.parameters())
        return layer_params

    def real_metrics_mapping(self):
        # mapping = {'first': '1st', 'second': '2rd', 'mix': 'mix', 'grad': 'grad', 'weight': 'weight'}
        mapping = {'first': 'new_1st', 'second': '2rd', 'mix': 'mix', 'grad': 'grad', 'weight': 'weight'}
        return mapping[self.config.task.prune.real_metrics]
