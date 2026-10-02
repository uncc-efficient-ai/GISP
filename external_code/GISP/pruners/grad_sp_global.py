# Includes attention routines adapted from Transformers (Apache-2.0).
# Modified for GISP iterative pruning and mask handling.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.

import copy
import logging
import math
import os
import time
import types
import warnings
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from transformers import Cache

from modules.eval.setup_eval import eval_ppl, eval_lm_eval
from tasks.pruning.pruners import Pruner
from torch import nn

from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *
import pickle
import threading

logger = logging.getLogger(__name__)


class grad_sp_global(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.gqa_mask_record = None

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['global_grad_sp']:
            self.global_grad_sp()

    def step(self):
        pass

    def global_grad_sp(self):
        head_dim = self.get_model().config.hidden_size // self.get_model().config.num_attention_heads
        head_num = self.get_model().config.num_attention_heads
        hidden_dim = self.get_model().config.hidden_size
        intermediate_size = self.get_model().config.intermediate_size
        layers = self.get_layers()
        metric_target = self.real_metrics_mapping()
        self.before_pruning()
        is_gqa = (self.get_model().config.num_key_value_heads < self.get_model().config.num_attention_heads)
        if is_gqa:
            repeat_times = self.get_model().config.num_attention_heads // self.get_model().config.num_key_value_heads
            original_kv_head_count = self.get_model().config.num_key_value_heads
            origin_function = {}
            self.gqa_mask_record = {}

            def monkey_patch_forward():
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    origin_function['func'] = type(attn_block).forward
                    if 'Llama' in self.config.model.name:
                        self.gqa_mask_record[index] = torch.ones(self.get_model().config.num_attention_heads,
                                                            device=self.get_model().device,
                                                            dtype=getattr(attn_block, self.layer_mapping['attn'][
                                                                'q_name']).weight.data.dtype)
                        attn_block.forward = types.MethodType(hooking_llama3(self.gqa_mask_record[index]),
                                                              attn_block)
                    elif 'Qwen' in self.config.model.name:
                        self.gqa_mask_record[index] = torch.ones(self.get_model().config.num_attention_heads,
                                                            device=self.get_model().device,
                                                            dtype=getattr(attn_block, self.layer_mapping['attn'][
                                                                'q_name']).weight.data.dtype)
                        attn_block.forward = types.MethodType(hooking_qwen2_5(self.gqa_mask_record[index]),
                                                              attn_block)
                    elif 'Mistral' in self.config.model.name:
                        self.gqa_mask_record[index] = torch.ones(self.get_model().config.num_attention_heads,
                                                            device=self.get_model().device,
                                                            dtype=getattr(attn_block, self.layer_mapping['attn'][
                                                                'q_name']).weight.data.dtype)
                        attn_block.forward = types.MethodType(hooking_mistral_0_3(self.gqa_mask_record[index]),
                                                              attn_block)

            def remove_patch_all():
                for index, l in enumerate(layers):
                    attn_block = getattr(l, self.layer_mapping['attn']['block'])
                    attn_block.forward = types.MethodType(origin_function['func'], attn_block)

            monkey_patch_forward()

        def obtain_info_iterative(layer_idxs):
            layers = self.get_layers()
            avg_loss = self.fill_information()
            grad_norm = {}
            weight_norm = {}
            grad_acc_norm = {}
            W_dicts = {}
            if isinstance(self.data_pos, dict):
                n_samples = sum(x.size(0) for x in self.data_pos.values())
            else:
                if self.config.task.prune.prune_dataset.type in ['open_domain']:
                    n_samples = self.config.task.prune.prune_dataset.n_samples
                else:
                    n_samples = self.data_pos.size(0)

            if isinstance(self.data_neg, dict):
                n_samples_neg = sum(x.size(0) for x in self.data_neg.values())
            else:
                if self.config.task.prune.prune_dataset.type in ['open_domain']:
                    n_samples_neg = self.config.task.prune.prune_dataset.n_samples
                else:
                    n_samples_neg = self.data_neg.size(0)

            for x in tqdm(range(layer_idxs, len(layers)), desc="Processing layers"):
                layer = layers[x]

                if self.config.task.prune.prune_modules in ['mha', 'all']:
                    subset = {}
                    subset.update(
                        {self.layer_mapping['attn']['q']: find_layers(layer)[self.layer_mapping['attn']['q']]})
                    subset.update(
                        {self.layer_mapping['attn']['k']: find_layers(layer)[self.layer_mapping['attn']['k']]})
                    subset.update(
                        {self.layer_mapping['attn']['v']: find_layers(layer)[self.layer_mapping['attn']['v']]})
                    subset.update(
                        {self.layer_mapping['attn']['o']: find_layers(layer)[self.layer_mapping['attn']['o']]})

                    for name in subset:
                        print(f"pruning layer {x} name {name}")
                        current_layer = subset[name]
                        norms = torch.norm(current_layer.weight)
                        weight_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        norms = torch.norm(current_layer.weight.grad)
                        grad_norm[f"{x}.{name}"] = norms.detach().clone().cpu()
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            norms = torch.norm(current_layer.weight.acc_grad)
                            grad_acc_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        W_metric_1st = current_layer.weight * (current_layer.weight.grad / n_samples)
                        W_metric_weight = current_layer.weight
                        W_metric_grad = (current_layer.weight.grad / n_samples)

                        if hasattr(current_layer.weight, 'grad_neg'):
                            W_metric_1st_neg = current_layer.weight * (current_layer.weight.grad_neg.to(current_layer.weight.device) / n_samples_neg)
                            W_metric_1st = W_metric_1st - W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = current_layer.weight * current_layer.weight.acc_grad * current_layer.weight
                            W_metric_mix = W_metric_1st - 0.5 * current_layer.weight * current_layer.weight.acc_grad * current_layer.weight

                        W_metric_1st = W_metric_1st.abs()
                        W_metric_weight = W_metric_weight.abs()
                        W_metric_grad = W_metric_grad.abs()

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.abs()

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.abs()
                            W_metric_mix = W_metric_mix.abs()

                        if name in [self.layer_mapping['attn']['o']]:
                            W_metric_1st = W_metric_1st.t()
                            W_metric_weight = W_metric_weight.t()
                            W_metric_grad = W_metric_grad.t()

                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_metric_1st_neg = W_metric_1st_neg.t()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_metric_second = W_metric_second.t()
                                W_metric_mix = W_metric_mix.t()

                        if is_gqa and name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                            continue

                        W_metric_1st = W_metric_1st.reshape(W_metric_1st.shape[0] // head_dim, -1)
                        W_metric_1st = W_metric_1st.sum(1)
                        W_metric_weight = W_metric_weight.reshape(W_metric_weight.shape[0] // head_dim, -1)
                        W_metric_weight = W_metric_weight.sum(1)
                        W_metric_grad = W_metric_grad.reshape(W_metric_grad.shape[0] // head_dim, -1)
                        W_metric_grad = W_metric_grad.sum(1)

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.reshape(W_metric_1st_neg.shape[0] // head_dim, -1)
                        #     W_metric_1st_neg = W_metric_1st_neg.sum(1)

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.reshape(W_metric_second.shape[0] // head_dim, -1)
                            W_metric_second = W_metric_second.sum(1)
                            W_metric_mix = W_metric_mix.reshape(W_metric_mix.shape[0] // head_dim, -1)
                            W_metric_mix = W_metric_mix.sum(1)

                        if f"{x}.{self.layer_mapping['attn']['block']}" in W_dicts:
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                '1st'] += W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'weight'] += W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'grad'] += W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                            #         '1st_neg'] += W_metric_1st_neg.detach().clone().cpu()
                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    '2rd'] += W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    'mix'] += W_metric_mix.detach().clone().cpu()
                        else:
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"] = {}
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                '1st'] = W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'weight'] = W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                'grad'] = W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                            #         '1st_neg'] = W_metric_1st_neg.detach().clone().cpu()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    '2rd'] = W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['attn']['block']}"][
                                    'mix'] = W_metric_mix.detach().clone().cpu()
                        del W_metric_1st, W_metric_weight, W_metric_grad
                        if hasattr(current_layer.weight, 'grad_neg'):
                            del W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            del W_metric_second, W_metric_mix

                if self.config.task.prune.prune_modules in ['mlp', 'all']:
                    subset = {}
                    subset.update({self.layer_mapping['mlp']['u']: find_layers(layer)[self.layer_mapping['mlp']['u']]})
                    if 'g' in self.layer_mapping['mlp']:
                        subset.update(
                            {self.layer_mapping['mlp']['g']: find_layers(layer)[self.layer_mapping['mlp']['g']]})
                    subset.update({self.layer_mapping['mlp']['d']: find_layers(layer)[self.layer_mapping['mlp']['d']]})

                    for name in subset:
                        print(f"pruning layer {x} name {name}")
                        current_layer = subset[name]
                        norms = torch.norm(current_layer.weight)
                        weight_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        norms = torch.norm(current_layer.weight.grad)
                        grad_norm[f"{x}.{name}"] = norms.detach().clone().cpu()
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            norms = torch.norm(current_layer.weight.acc_grad)
                            grad_acc_norm[f"{x}.{name}"] = norms.detach().clone().cpu()

                        W_metric_1st = current_layer.weight * (current_layer.weight.grad / n_samples)
                        W_metric_weight = current_layer.weight
                        W_metric_grad = (current_layer.weight.grad / n_samples)

                        if hasattr(current_layer.weight, 'grad_neg'):
                            W_metric_1st_neg = current_layer.weight * (current_layer.weight.grad_neg.to(current_layer.weight.device) / n_samples_neg)
                            W_metric_1st = W_metric_1st - W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = current_layer.weight * current_layer.weight.acc_grad * current_layer.weight
                            W_metric_mix = W_metric_1st - 0.5 * current_layer.weight * current_layer.weight.acc_grad * current_layer.weight

                        W_metric_1st = W_metric_1st.abs()
                        W_metric_weight = W_metric_weight.abs()
                        W_metric_grad = W_metric_grad.abs()

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.abs()

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.abs()
                            W_metric_mix = W_metric_mix.abs()

                        if name in [self.layer_mapping['mlp']['d']]:
                            W_metric_1st = W_metric_1st.t()
                            W_metric_weight = W_metric_weight.t()
                            W_metric_grad = W_metric_grad.t()

                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_metric_1st_neg = W_metric_1st_neg.t()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_metric_second = W_metric_second.t()
                                W_metric_mix = W_metric_mix.t()

                        W_metric_1st = W_metric_1st.sum(1)
                        W_metric_weight = W_metric_weight.sum(1)
                        W_metric_grad = W_metric_grad.sum(1)

                        # if hasattr(current_layer.weight, 'grad_neg'):
                        #     W_metric_1st_neg = W_metric_1st_neg.sum(1)

                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            W_metric_second = W_metric_second.sum(1)
                            W_metric_mix = W_metric_mix.sum(1)

                        if f"{x}.{self.layer_mapping['mlp']['block']}" in W_dicts:
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                '1st'] += W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'weight'] += W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'grad'] += W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                            #         '1st_neg'] += W_metric_1st_neg.detach().clone().cpu()

                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    '2rd'] += W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    'mix'] += W_metric_mix.detach().clone().cpu()

                        else:
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"] = {}
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                '1st'] = W_metric_1st.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'weight'] = W_metric_weight.detach().clone().cpu()
                            W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                'grad'] = W_metric_grad.detach().clone().cpu()
                            # if hasattr(current_layer.weight, 'grad_neg'):
                            #     W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                            #         '1st_neg'] = W_metric_1st_neg.detach().clone().cpu()
                            if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    '2rd'] = W_metric_second.detach().clone().cpu()
                                W_dicts[f"{x}.{self.layer_mapping['mlp']['block']}"][
                                    'mix'] = W_metric_mix.detach().clone().cpu()
                        del W_metric_1st, W_metric_weight, W_metric_grad
                        if hasattr(current_layer.weight, 'grad_neg'):
                            del W_metric_1st_neg
                        if self.config.task.prune.taylor in ['param_second', 'param_mix']:
                            del W_metric_second, W_metric_mix

                torch.cuda.empty_cache()
            if not self.config.task.prune.prune_separate:
                for k, v in W_dicts.items():
                    if f"{self.layer_mapping['mlp']['block']}" in k:
                        structural_size = intermediate_size
                        if 'g' in self.layer_mapping['mlp']:
                            structural_size = structural_size * 3
                        else:
                            structural_size = structural_size * 2
                    elif f"{self.layer_mapping['attn']['block']}" in k:
                        structural_size = 4 * head_dim * hidden_dim
                    else:
                        raise ValueError
                    for v_key, v_value in v.items():
                        v[v_key] = v_value / structural_size
                for k, v in W_dicts.items():
                    v['new_1st'] = v['1st']
                    # first = v['1st'].float()
                    # first_neg = v['1st_neg'].float()
                    # v['new_1st'] = first
                    # invalid_division = (first_neg == 0) & (first != 0)
                    # if invalid_division.any():
                    #     raise ValueError
                    # result = torch.zeros_like(first)
                    # 只在 first_neg 不为零的位置执行除法
                    # valid_mask = first_neg != 0
                    # result[valid_mask] = first[valid_mask] / first_neg[valid_mask]
                    # 赋值给新键
                    # v['new_1st_ratio'] = result
                    # v['new_1st'] = v['new_1st_ratio'] * first
            return W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss

        def prune_oneshot(W_dicts, inter_layer_imp, tr, bypassed_blocks):
            self.before_pruning_step()
            current_mask_record = {}
            layers = self.get_layers()
            # target scope: layer_idxs to len(layers)
            metrics_mha_std = torch.tensor([])
            metrics_mlp_std = torch.tensor([])
            # tr = self.min_max_scope_limit(tr)
            # self.model = self.get_model().to('cpu')
            torch.cuda.empty_cache()
            start_time = time.time()
            global_start = time.time()
            for k, v in W_dicts.items():
                name = k.split('.')
                if int(name[0]) not in range(int(len(layers) * 0.1), len(layers) - 1):
                    if self.config.task.prune.prune_skip:
                        new_v = torch.zeros_like(v[metric_target])
                        new_v = torch.fill(new_v, torch.inf)
                        if name[1] in [self.layer_mapping['attn']['block']]:
                            metrics_mha_std = torch.cat((metrics_mha_std, new_v), dim=0)
                        else:
                            metrics_mlp_std = torch.cat((metrics_mlp_std, new_v), dim=0)
                        continue
                if f"{name[0]}.{name[1]}" in bypassed_blocks:
                    new_v = v[metric_target].clone()
                    non_zero_mask = (new_v != 0)
                    new_v[non_zero_mask] = torch.inf
                    if name[1] in [self.layer_mapping['attn']['block']]:
                        metrics_mha_std = torch.cat((metrics_mha_std, new_v), dim=0)
                    else:
                        metrics_mlp_std = torch.cat((metrics_mlp_std, new_v), dim=0)
                    continue
                if name[1] in [self.layer_mapping['attn']['block']]:
                    if self.config.task.prune.prune_modules in ['mha', 'all']:
                        metrics_mha_std = torch.cat((metrics_mha_std, v[metric_target]), dim=0)
                else:
                    if self.config.task.prune.prune_modules in ['mlp', 'all']:
                        metrics_mlp_std = torch.cat((metrics_mlp_std, v[metric_target]), dim=0)

            pruning_ratio_record_mha = {}
            pruning_ratio_record_mlp = {}
            if self.config.task.prune.prune_separate:
                logger.info(f'cat metrics time cost: {time.time() - start_time}')
                if self.config.task.prune.prune_modules in ['mha', 'all']:
                    start_time = time.time()
                    metrics_mha_std = metrics_mha_std.to('cuda')
                    metrics_mha_std = metrics_mha_std + torch.abs(torch.min(metrics_mha_std))
                    # for z in range(len(layers)):
                    #     metrics_mha_std[head_num * z:(z + 1) * head_num] *= (inter_layer_imp[z])
                    origin_size = metrics_mha_std.size()
                    metrics_mha_std = metrics_mha_std.view(-1)
                    N = metrics_mha_std.numel()
                    k_mha = int(N * (1 - tr))
                    k_mha = N - k_mha + 1
                    metrics_mha_std = metrics_mha_std.to('cpu')
                    threshold, _ = torch.kthvalue(metrics_mha_std, k=k_mha)
                    threshold = threshold.to('cuda')
                    metrics_mha_std = metrics_mha_std.to('cuda')

                    W_mask_mha = (metrics_mha_std <= threshold)
                    W_mask_mha = W_mask_mha.view(origin_size)
                    W_mask_mha = W_mask_mha.to('cpu')
                    logger.info(f'threshold for mha: {threshold}')
                    logger.info(f'total sparsity for mha: {W_mask_mha.sum() / N}')
                    logger.info(f'kth & generate mask for mha time cost: {time.time() - start_time}')

                    mha_size = head_num * head_dim

                    for i in range(len(layers)):
                        if is_gqa:
                            submask_layer = W_mask_mha[i * head_num: (i + 1) * head_num]
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer
                            submask_v = submask_layer
                            submask_o = submask_layer.repeat_interleave(head_dim)
                        else:
                            submask_layer = W_mask_mha[i * head_num: (i + 1) * head_num]
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer.repeat_interleave(head_dim)
                            submask_v = submask_layer.repeat_interleave(head_dim)
                            submask_o = submask_layer.repeat_interleave(head_dim)
                        for name, vis_name, submask in zip(
                                [self.layer_mapping['attn']['q'], self.layer_mapping['attn']['k'],
                                 self.layer_mapping['attn']['v'], self.layer_mapping['attn']['o']],
                                [self.layer_mapping['attn']['q_name'],
                                 self.layer_mapping['attn']['k_name'],
                                 self.layer_mapping['attn']['v_name'],
                                 self.layer_mapping['attn']['o_name']], [
                                    submask_q, submask_k, submask_v, submask_o]):
                            self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_mha_std, threshold)
                            if name in [self.layer_mapping['attn']['o']]:
                                find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                            else:
                                if is_gqa:
                                    if name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                                        self.gqa_mask_record[i].data[submask] = 0
                                    else:
                                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                            if vis_name not in pruning_ratio_record_mha:
                                pruning_ratio_record_mha[vis_name] = [submask.sum() / mha_size]
                            else:
                                pruning_ratio_record_mha[vis_name].append(submask.sum() / mha_size)
                    del metrics_mha_std
                    del W_mask_mha

                torch.cuda.empty_cache()

                if self.config.task.prune.prune_modules in ['mlp', 'all']:
                    start_time = time.time()
                    metrics_mlp_std = metrics_mlp_std.to('cuda')
                    metrics_mlp_std = metrics_mlp_std + torch.abs(torch.min(metrics_mlp_std))
                    # for z in range(len(layers)):
                    #     metrics_mlp_std[z * intermediate_size:(z + 1) * intermediate_size] *= (inter_layer_imp[z])
                    origin_size = metrics_mlp_std.size()
                    metrics_mlp_std = metrics_mlp_std.view(-1)
                    N = metrics_mlp_std.numel()
                    k_mlp = int(N * (1 - tr))
                    k_mlp = N - k_mlp + 1
                    metrics_mlp_std = metrics_mlp_std.to('cpu')
                    mlp_threshold, ndx = torch.kthvalue(metrics_mlp_std, k=k_mlp)
                    mlp_threshold = mlp_threshold.to('cuda')
                    metrics_mlp_std = metrics_mlp_std.to('cuda')
                    W_mask_mlp = (metrics_mlp_std <= mlp_threshold)
                    W_mask_mlp = W_mask_mlp.view(origin_size)
                    W_mask_mlp = W_mask_mlp.to('cpu')
                    logger.info(f'threshold for mlp: {mlp_threshold}')
                    logger.info(f'total sparsity for mlp: {W_mask_mlp.sum() / N}')
                    logger.info(f'kth & generate mask for mlp time cost: {time.time() - start_time}')

                    mlp_size = intermediate_size

                    for i in range(len(layers)):
                        if 'g' in self.layer_mapping['mlp']:
                            submask_layer = W_mask_mlp[i * intermediate_size:(i + 1) * intermediate_size]
                            submask_u = submask_layer
                            submask_g = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'], self.layer_mapping['mlp']['g'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['g_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_g, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_mlp_std,
                                                         mlp_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                        else:
                            submask_layer = W_mask_mlp[i * intermediate_size:(i + 1) * intermediate_size]
                            submask_u = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_mlp_std,
                                                         mlp_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                    del metrics_mlp_std
                    del W_mask_mlp
            else:
                if self.config.task.prune.prune_modules in ['all']:
                    size_mha = metrics_mha_std.size()
                    size_mlp = metrics_mlp_std.size()
                    metrics_all_std = torch.cat((metrics_mha_std, metrics_mlp_std), dim=0)
                    logger.info(f'cat metrics time cost: {time.time() - start_time}')

                    start_time = time.time()
                    metrics_all_std = metrics_all_std.to('cuda')
                    metrics_all_std = metrics_all_std + torch.abs(torch.min(metrics_all_std))
                    # for z in range(len(layers)):
                    #     metrics_mlp_std[z * intermediate_size:(z + 1) * intermediate_size] *= (inter_layer_imp[z])
                    origin_size = metrics_all_std.size()
                    metrics_all_std = metrics_all_std.view(-1)
                    N = metrics_all_std.numel()
                    k_all = int(N * (1 - tr))
                    k_all = N - k_all + 1
                    metrics_all_std = metrics_all_std.to('cpu')
                    all_threshold, ndx = torch.kthvalue(metrics_all_std, k=k_all)
                    all_threshold = all_threshold.to('cuda')
                    metrics_all_std = metrics_all_std.to('cuda')
                    W_mask_all = (metrics_all_std <= all_threshold)
                    W_mask_all = W_mask_all.view(origin_size)
                    W_mask_all = W_mask_all.to('cpu')
                    logger.info(f'threshold for all: {all_threshold}')
                    logger.info(f'total sparsity for all: {W_mask_all.sum() / N}')
                    logger.info(f'kth & generate mask for all time cost: {time.time() - start_time}')

                    W_mask_mha = W_mask_all[:size_mha[0]]
                    W_mask_mlp = W_mask_all[size_mha[0]:size_mha[0] + size_mlp[0]]
                    mha_ratio = W_mask_all[:size_mha[0]].sum() / size_mha[0]
                    mlp_ratio = W_mask_all[size_mha[0]:size_mha[0] + size_mlp[0]].sum() / size_mlp[0]
                    del metrics_mha_std
                    del metrics_mlp_std
                    mha_size = head_num * head_dim

                    for i in range(len(layers)):
                        submask_layer = W_mask_mha[i * head_num: (i + 1) * head_num]
                        current_mask_record[f"{i}.{self.layer_mapping['attn']['block']}"] = submask_layer
                        if is_gqa:
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer
                            submask_v = submask_layer
                            submask_o = submask_layer.repeat_interleave(head_dim)
                        else:
                            submask_q = submask_layer.repeat_interleave(head_dim)
                            submask_k = submask_layer.repeat_interleave(head_dim)
                            submask_v = submask_layer.repeat_interleave(head_dim)
                            submask_o = submask_layer.repeat_interleave(head_dim)

                        for name, vis_name, submask in zip(
                                [self.layer_mapping['attn']['q'], self.layer_mapping['attn']['k'],
                                 self.layer_mapping['attn']['v'], self.layer_mapping['attn']['o']],
                                [self.layer_mapping['attn']['q_name'],
                                 self.layer_mapping['attn']['k_name'],
                                 self.layer_mapping['attn']['v_name'],
                                 self.layer_mapping['attn']['o_name']], [
                                    submask_q, submask_k, submask_v, submask_o]):
                            self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_all_std,
                                                     all_threshold)
                            if name in [self.layer_mapping['attn']['o']]:
                                find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                            else:
                                if is_gqa:
                                    if name in [self.layer_mapping['attn']['k'], self.layer_mapping['attn']['v']]:
                                        self.gqa_mask_record[i].data[submask] = 0
                                    else:
                                        find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                            if vis_name not in pruning_ratio_record_mha:
                                pruning_ratio_record_mha[vis_name] = [submask.sum() / mha_size]
                            else:
                                pruning_ratio_record_mha[vis_name].append(submask.sum() / mha_size)
                    del W_mask_mha
                    mlp_size = intermediate_size
                    for i in range(len(layers)):
                        submask_layer = W_mask_mlp[i * intermediate_size:(i + 1) * intermediate_size]
                        current_mask_record[f"{i}.{self.layer_mapping['mlp']['block']}"] = submask_layer
                        if 'g' in self.layer_mapping['mlp']:
                            submask_u = submask_layer
                            submask_g = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'], self.layer_mapping['mlp']['g'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['g_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_g, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_all_std,
                                                         all_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero
                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                        else:
                            submask_u = submask_layer
                            submask_d = submask_layer
                            for name, vis_name, submask in zip(
                                    [self.layer_mapping['mlp']['u'],
                                     self.layer_mapping['mlp']['d']],
                                    [self.layer_mapping['mlp']['u_name'],
                                     self.layer_mapping['mlp']['d_name']], [
                                        submask_u, submask_d]):
                                self.during_pruning_step(find_layers(layers[i])[name], submask, metrics_all_std,
                                                         all_threshold)
                                if name in [self.layer_mapping['mlp']['d']]:
                                    find_layers(layers[i])[name].weight.data[:, submask] = 0  ## set weights to zero
                                else:
                                    find_layers(layers[i])[name].weight.data[submask] = 0  ## set weights to zero

                                if vis_name not in pruning_ratio_record_mlp:
                                    pruning_ratio_record_mlp[vis_name] = [submask.sum() / mlp_size]
                                else:
                                    pruning_ratio_record_mlp[vis_name].append(submask.sum() / mlp_size)
                    del W_mask_mlp
                    del W_mask_all
                    del metrics_all_std
                else:
                    raise ValueError()
            logger.info(f'global time cost: {time.time() - global_start}')
            # self.model = self.get_model().to('cuda')
            torch.cuda.empty_cache()
            self.save_helper.append("actual_mask", current_mask_record)
            if self.config.task.prune.prune_separate:
                return pruning_ratio_record_mha, pruning_ratio_record_mlp
            else:
                return pruning_ratio_record_mha, pruning_ratio_record_mlp, mha_ratio, mlp_ratio

        def check_grad_change(cur_W_dicts, historical_info, bypassed_blocks):
            if historical_info == {}:
                return None, False, historical_info
            for k, v in cur_W_dicts.items():
                if k in bypassed_blocks:
                    continue
                prev_step_information = historical_info[k]
                # compare the gradient change with expectation
                prev_metric_target = prev_step_information[metric_target][-1]
                cur_metric_target = v[metric_target].mean()
                if cur_metric_target > self.config.task.prune.boundary_threshold * prev_metric_target:
                    logger.info(
                        f"{metric_target} jump of {k} detected. {metric_target} change from {prev_metric_target} to {cur_metric_target}.")
                    return k, True, historical_info
            return None, False, historical_info

        def fulfill_historical_info(cur_W_dicts, historical_info):
            for k, v in cur_W_dicts.items():
                if k not in historical_info:
                    historical_info[k] = {}
                    for type, score in v.items():
                        historical_info[k][type] = []
                for type, score in v.items():
                    historical_info[k][type].append(score.mean())
            return historical_info

        def remove_historical_info(historical_info):
            for k, v in historical_info.items():
                for type, score in v.items():
                    historical_info[k][type].pop()
            return historical_info

        def fulfill_pruning_ratio_records(ratio_mha, ratio_mlp, pruning_ratio_records, bypassed_blocks):
            if ratio_mha is not None and ratio_mlp is not None:
                for k, v in ratio_mha.items():
                    for layer_index, pruned_ratio in enumerate(v):
                        if f'{str(layer_index)}.{k}' not in pruning_ratio_records:
                            pruning_ratio_records[f'{str(layer_index)}.{k}'] = [torch.tensor(0)]
                        pruning_ratio_records[f'{str(layer_index)}.{k}'].append(pruned_ratio)
                for k, v in ratio_mlp.items():
                    for layer_index, pruned_ratio in enumerate(v):
                        if f'{str(layer_index)}.{k}' not in pruning_ratio_records:
                            pruning_ratio_records[f'{str(layer_index)}.{k}'] = [torch.tensor(0)]
                        pruning_ratio_records[f'{str(layer_index)}.{k}'].append(pruned_ratio)
            return pruning_ratio_records

        def remove_pruning_ratio_records(pruning_ratio_records):
            for k, v in pruning_ratio_records.items():
                if len(v) > 0:
                    pruning_ratio_records[k].pop()
            return pruning_ratio_records

        def back_up_model():
            self.get_model().to('cpu')
            backup_model = copy.deepcopy(self.model)
            self.get_model().to('cuda')
            return backup_model

        def restore_model(backup_model):
            self.get_model().to('cpu')
            self.model = copy.deepcopy(backup_model)
            self.get_model().to('cuda')

        def evaluate_intermediate(eval_intermediate_lists, intermediate_index, prev_sparsity, prev_model, cur_sparsity,
                                  cur_model):
            if intermediate_index >= len(eval_intermediate_lists):
                return False
            next_intermediate_sparsity = eval_intermediate_lists[intermediate_index]
            flag = prev_sparsity <= next_intermediate_sparsity <= cur_sparsity
            if flag:
                distance_prev = abs(prev_sparsity - next_intermediate_sparsity)
                distance_cur = abs(cur_sparsity - next_intermediate_sparsity)
                if distance_prev < distance_cur:
                    cur_model = cur_model.to('cpu')
                    prev_model = prev_model.to('cuda')
                    eval_ppl(prev_model, self.tokenizer, save=True,
                             save_path=f'{self.config.task.output_folder}/sp_{prev_sparsity}_ppl.pth')
                    eval_lm_eval(prev_model, self.tokenizer, self.config, f'sp_{prev_sparsity}_lm_eval',
                                 quick=False)
                    prev_model = prev_model.to('cpu')
                    cur_model = cur_model.to('cuda')
                else:
                    eval_ppl(cur_model, self.tokenizer, save=True,
                             save_path=f'{self.config.task.output_folder}/sp_{cur_sparsity}_ppl.pth')
                    eval_lm_eval(cur_model, self.tokenizer, self.config, f'sp_{cur_sparsity}_lm_eval',
                                 quick=False)
            return flag

        def prune_iterative(iteration):
            rollback_enabled = self.config.task.prune.rollback
            eval_intermediate = self.config.task.prune.eval_intermediate
            rollback_allowed = True
            bypassed_blocks = []
            ratios = self.ratio_scheduling(iteration)
            show_diagram_list(ratios,
                              "ratio scheduling",
                              save_location=self.config.task.output_folder)
            backup_model = None
            backup_sparsity = None
            intermediate_index = 0
            buffer_sparsity = self.check_sparsity(real_pruning=False)
            if rollback_enabled or eval_intermediate:
                backup_model = back_up_model()
                backup_sparsity = buffer_sparsity
            historical_info = {}
            pruning_ratio_records = {}
            iter_index = 0
            ratio_mha, ratio_mlp = None, None
            while iter_index < len(ratios):
                if len(bypassed_blocks) == 80:
                    break
                r = ratios[iter_index]
                W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss = obtain_info_iterative(0)
                current_sparsity = self.check_sparsity(real_pruning=False)

                self.save_files(current_sparsity, is_gqa, W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss,
                                pruning_ratio_records, historical_info,
                                None if not is_gqa else self.gqa_mask_record)
                if eval_intermediate:
                    flag = evaluate_intermediate(eval_intermediate, intermediate_index, backup_sparsity, backup_model,
                                                 current_sparsity,
                                                 self.get_wrapped_model())
                    if flag:
                        intermediate_index += 1
                bypassed_block, detected, historical_info = check_grad_change(W_dicts, historical_info, bypassed_blocks)
                # if detected and not rollback_allowed:
                #     logger.info(f"Trying to rollback, but we just rollback once!")
                if rollback_enabled and detected and rollback_allowed:
                    # only thing we know is that we can not prune this block anymore.
                    logger.info(f"Rollback from {r} to {ratios[max(0, iter_index - 1)]}.")
                    self.on_rollback_happened()
                    bypassed_blocks.append(bypassed_block)
                    iter_index = max(0, iter_index - 1)
                    self.delete_files(current_sparsity, is_gqa)
                    self.evaluation(r, f'{self.config.task.output_folder}/rollback_sp_{current_sparsity}_ppl.pth')
                    restore_model(backup_model)
                    historical_info = remove_historical_info(historical_info)
                    pruning_ratio_records = remove_pruning_ratio_records(pruning_ratio_records)
                    rollback_allowed = False
                    continue
                else:
                    rollback_allowed = True
                    if current_sparsity != buffer_sparsity:
                        historical_info = fulfill_historical_info(W_dicts, historical_info)
                    else:
                        logger.info('No add-on sparsity in this step. Skip updating historical importance records.')
                    iter_index += 1
                self.get_model().zero_grad()
                for param in self.get_model().parameters():
                    param.requires_grad_(False)
                if rollback_enabled or eval_intermediate:
                    backup_model = back_up_model()
                    backup_sparsity = current_sparsity
                if current_sparsity != buffer_sparsity:
                    pruning_ratio_records = fulfill_pruning_ratio_records(ratio_mha, ratio_mlp, pruning_ratio_records,
                                                                          bypassed_blocks)
                else:
                    logger.info('No add-on sparsity in this step. Skip updating pruning ratio records.')

                if self.config.task.prune.prune_separate:
                    ratio_mha, ratio_mlp = prune_oneshot(W_dicts, None, r, bypassed_blocks)
                else:
                    ratio_mha, ratio_mlp, mha_block_ratio, mlp_block_ratio = prune_oneshot(W_dicts, None, r,
                                                                                           bypassed_blocks)
                    current_sparsity = self.check_sparsity(real_pruning=False)
                    self.save_helper.append("block_wise_ratio", {'mha': mha_block_ratio, 'mlp': mlp_block_ratio})
                self.after_pruning_step(r, ratio_mlp, ratio_mha)

        if self.config.task.prune.iterative:
            prune_iterative(self.config.task.prune.iteration)
        else:
            prune_iterative(1)
        if is_gqa:
            self.gqa_mask_record = self.gqa_mask_record
        self.finishing_pruning(real_pruning=False)


    def save_files(self, current_sparsity, is_gqa, W_dicts, grad_norm, weight_norm, grad_acc_norm, avg_loss,
                   pruning_ratio_records, historical_info, gqa_mask_record=None):

        self.save_helper.append("w_dicts", W_dicts)
        self.save_helper.append("grad_norm", grad_norm)
        self.save_helper.append("weight_norm", weight_norm)
        self.save_helper.append("grad_acc_norm", grad_acc_norm)
        self.save_helper.append("avg_loss", avg_loss)
        self.save_helper.append("pruning_ratio_records", pruning_ratio_records)
        self.save_helper.append("historical_info", historical_info)
        if is_gqa:
            gqa_mask_record_cpu = {}
            for key, value in gqa_mask_record.items():
                if isinstance(value, torch.Tensor):
                    gqa_mask_record_cpu[key] = value.cpu()
                else:
                    gqa_mask_record_cpu[key] = value
            self.save_helper.append("gqa_mask_record", gqa_mask_record_cpu)
        self.save_helper.save(current_sparsity)

    def delete_files(self, current_sparsity, is_gqa=False):
        self.save_helper.delete(current_sparsity)

    def after_pruning_step(self, r, ratio_mlp, ratio_mha):
        r = self.check_sparsity(real_pruning=False)
        available_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {available_gpus}")
        base_model = self.get_wrapped_model()
        if available_gpus >= 2:
            def run_eval_0():
                model_gpu0 = base_model.to("cuda:0")
                config_clone = copy.deepcopy(self.config)
                # config_clone.evaluation.lm_eval_options.quick_tasks = ["hellaswag"]
                config_clone.evaluation.lm_eval_options.quick_tasks = ["arc_challenge"]
                config_clone.evaluation.lm_eval_options.batch_size = 32
                eval_lm_eval(model_gpu0, self.tokenizer, config_clone, f'sp_{r}_lm_eval_quick_0', quick=True)

            def run_eval_1():
                # arc_challenge 4687 2min
                # arc_easy 9501 1min
                # openbookqa 2000 30sec
                # winogrande 2534 12sec
                # boolq 6540 2min19
                # piqa 3676 2min
                # hellaswag 40168 10min
                model_gpu1 = copy.deepcopy(base_model).to("cuda:1")
                config_clone = copy.deepcopy(self.config)
                config_clone.evaluation.lm_eval_options.quick_tasks = ["winogrande", "piqa", "openbookqa", "arc_easy",
                                                                       "boolq",
                                                                       "arc_challenge"]
                config_clone.evaluation.lm_eval_options.batch_size = 32
                eval_ppl(model_gpu1, self.tokenizer, save=True,
                         save_path=f'{self.config.task.output_folder}/sp_{r}_ppl.pth')
                eval_lm_eval(model_gpu1, self.tokenizer, config_clone, f'sp_{r}_lm_eval_quick_1', quick=True)

            # t1 = threading.Thread(target=run_eval_0)
            # t2 = threading.Thread(target=run_eval_1)
            # t1.start()
            # t2.start()
            # t1.join()
            # t2.join()
            torch.cuda.empty_cache()
        else:
            logger.info("Only one GPU available, running evaluations sequentially")
            # eval_ppl(self.get_wrapped_model(), self.tokenizer, save=True,
            #          save_path=f'{self.config.task.output_folder}/sp_{r}_ppl.pth')
            # eval_lm_eval(self.get_wrapped_model(), self.tokenizer, self.config, f'sp_{r}_lm_eval_quick_0', quick=True)
        logger.info(f"ratio mlp: {ratio_mlp}")
        logger.info(f"ratio mha: {ratio_mha}")
        show_diagram_dict(ratio_mlp,
                          f"structure-wise grad global={r} (mlp) separate",
                          save_location=self.config.task.output_folder)
        show_diagram_dict(ratio_mha,
                          f"structure-wise grad global={r} (mha) separate",
                          save_location=self.config.task.output_folder)

    def on_rollback_happened(self):
        return

    def get_imps(self):
        return self.W_metrics

    def finishing_pruning(self, real_pruning=True):
        super().finishing_pruning(real_pruning)
        if self.gqa_mask_record is not None:
            torch.save(self.gqa_mask_record, f'{self.config.task.output_folder}/sp_gqa_mask.pth')


def hooking_llama3(mask, real_prune=False):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`, *optional*):
                Deprecated and unused.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        reshaped_mask = mask.view(1, key_states.shape[1], 1, 1)
        reshaped_mask = reshaped_mask.to(key_states.device)
        reshaped_mask = reshaped_mask.to(key_states.dtype)
        if real_prune:
            keep_heads = torch.where(mask)[0]
            key_states = key_states[:, keep_heads]
            value_states = value_states[:, keep_heads]
        else:
            key_states = key_states * reshaped_mask
            value_states = value_states * reshaped_mask

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def hooking_qwen2_5(mask, real_prune=False):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        reshaped_mask = mask.view(1, key_states.shape[1], 1, 1)
        reshaped_mask = reshaped_mask.to(key_states.device)
        reshaped_mask = reshaped_mask.to(key_states.dtype)
        if real_prune:
            keep_heads = torch.where(mask)[0]
            key_states = key_states[:, keep_heads]
            value_states = value_states[:, keep_heads]
        else:
            key_states = key_states * reshaped_mask
            value_states = value_states * reshaped_mask

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward


def hooking_mistral_0_3(mask, real_prune=False):
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q (`torch.Tensor`): The query tensor.
            k (`torch.Tensor`): The key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            position_ids (`torch.Tensor`):
                The position indices of the tokens corresponding to the query and key tensors. For example, this can be
                used to pass offsetted position ids when working with a KV-cache.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos[position_ids].unsqueeze(unsqueeze_dim)
        sin = sin[position_ids].unsqueeze(unsqueeze_dim)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
        num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
        """
        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        reshaped_mask = mask.view(1, key_states.shape[1], 1, 1)
        reshaped_mask = reshaped_mask.to(key_states.device)
        reshaped_mask = reshaped_mask.to(key_states.dtype)
        if real_prune:
            keep_heads = torch.where(mask)[0]
            key_states = key_states[:, keep_heads]
            value_states = value_states[:, keep_heads]
        else:
            key_states = key_states * reshaped_mask
            value_states = value_states * reshaped_mask

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    return forward
