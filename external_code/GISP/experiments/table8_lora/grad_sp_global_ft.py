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

from external_code.GISP.pruners.non_uniform_pruner import non_uniform_pruner
from external_code.GISP.pruners.layerwrapper import *
from .train import main
from external_code.GISP.pruners.utils import *
import pickle
import threading

logger = logging.getLogger(__name__)


class grad_sp_global_ft(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self.gqa_mask_record = None

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['restore_then_eval']:
            self.before_pruning()
            self.restore_to_prune(target_checkpoint_path=self.config.task.prune.restore_config.checkpoint_path)
            self.get_model_config().use_cache = self.use_cache
            torch.cuda.empty_cache()
            r = self.check_sparsity(True)
            eval_ppl(self.get_wrapped_model(), self.tokenizer, save=True,
                     save_path=f'{self.config.task.output_folder}/sp_{r}_ppl.pth')
            eval_lm_eval(self.get_wrapped_model(), self.tokenizer, self.config,
                         f'sp_{r}_lm_eval_before_ft', quick=False)
            self.model = main(self.model, self.tokenizer, f'{self.config.task.output_folder}/ft',
                              ['--data_module', 'modules.data.data_prune_ds',
                               '--no_trim_pad'
                               ])
            eval_lm_eval(self.get_wrapped_model(), self.tokenizer, self.config,
                         f'sp_{r}_lm_eval_after_ft', quick=False)

    def step(self):
        pass

    def restore_to_prune(self, target_sparsity=None, target_checkpoint_path=None, extra_path=None):
        layers = self.get_layers()
        head_dim = self.get_model().config.hidden_size // self.get_model().config.num_attention_heads
        is_gqa = (self.get_model().config.num_key_value_heads < self.get_model().config.num_attention_heads)
        self.save_helper.clean()
        if target_sparsity is not None:
            checkpoint = self.save_helper.load_sparsity(target_sparsity, extra_path=extra_path)
        elif target_checkpoint_path is not None:
            checkpoint = self.save_helper.load(target_checkpoint_path)
        else:
            raise ValueError

        layer_mapping=self.layer_mapping

        if "actual_mask" not in checkpoint:
            # dense model
            pass
        else:
            mask = checkpoint["actual_mask"]
            for i in range(len(layers)):
                if f"{i}.{layer_mapping['attn']['block']}" in mask:
                    submask_layer = mask[f"{i}.{layer_mapping['attn']['block']}"]  # head 级 mask

                    # -------- 非 GQA：直接结构化剪枝 --------
                    if not is_gqa:
                        # 先更新这一层的 head 数
                        attn_block = getattr(layers[i], layer_mapping['attn']['block'])
                        if submask_layer.dtype == torch.bool:
                            head_keep = ~submask_layer
                        else:
                            head_keep = torch.ones_like(submask_layer, dtype=torch.bool, device=submask_layer.device)
                            head_keep[submask_layer] = False
                        new_num_heads = int(head_keep.sum().item())
                        if hasattr(attn_block, "num_heads"):
                            attn_block.num_heads = new_num_heads
                        if hasattr(attn_block, "num_attention_heads"):
                            attn_block.num_attention_heads = new_num_heads
                        if hasattr(attn_block, "num_key_value_heads"):
                            attn_block.num_key_value_heads = new_num_heads
                        if hasattr(attn_block, "hidden_size"):
                            attn_block.hidden_size = new_num_heads * head_dim
                        # 原来一样：head mask 展开到 channel 级
                        submask_q = submask_layer.repeat_interleave(head_dim)
                        submask_k = submask_layer.repeat_interleave(head_dim)
                        submask_v = submask_layer.repeat_interleave(head_dim)
                        submask_o = submask_layer.repeat_interleave(head_dim)

                        for name, vis_name, submask in zip(
                                [layer_mapping['attn']['q'], layer_mapping['attn']['k'],
                                 layer_mapping['attn']['v'], layer_mapping['attn']['o']],
                                [layer_mapping['attn']['q_name'],
                                 layer_mapping['attn']['k_name'],
                                 layer_mapping['attn']['v_name'],
                                 layer_mapping['attn']['o_name']],
                                [submask_q, submask_k, submask_v, submask_o]):

                            linear = find_layers(layers[i])[name]
                            weight = linear.weight.data

                            if name in [layer_mapping['attn']['o']]:
                                # 原来是：weight.data[:, submask] = 0  （按列置零）
                                # 现在：按列裁剪 -> in_features 变小
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
                                    keep[submask] = False
                                new_in = int(keep.sum().item())
                                new_weight = weight[:, keep].clone()
                                linear.in_features = new_in
                                linear.weight = torch.nn.Parameter(new_weight)
                                # O 的 bias 维度 = out_features，不变
                            else:
                                # 原来是：weight.data[submask] = 0  （按行置零）
                                # 现在：按行裁剪 -> out_features 变小
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
                                    keep[submask] = False
                                new_out = int(keep.sum().item())
                                new_weight = weight[keep].clone()
                                linear.out_features = new_out
                                linear.weight = torch.nn.Parameter(new_weight)
                                if linear.bias is not None:
                                    new_bias = linear.bias.data[keep].clone()
                                    linear.bias = torch.nn.Parameter(new_bias)

                    # -------- GQA：保留你原来的「置零 + hook」逻辑 --------
                    else:
                        submask_q = submask_layer.repeat_interleave(head_dim)
                        submask_k = submask_layer
                        submask_v = submask_layer
                        submask_o = submask_layer.repeat_interleave(head_dim)

                        for name, vis_name, submask in zip(
                                [layer_mapping['attn']['q'], layer_mapping['attn']['k'],
                                 layer_mapping['attn']['v'], layer_mapping['attn']['o']],
                                [layer_mapping['attn']['q_name'],
                                 layer_mapping['attn']['k_name'],
                                 layer_mapping['attn']['v_name'],
                                 layer_mapping['attn']['o_name']],
                                [submask_q, submask_k, submask_v, submask_o]):

                            if name == layer_mapping['attn']['q']:
                                linear = find_layers(layers[i])[name]
                                weight = linear.weight.data  # [out_features, in_features]

                                # 原来的 submask 表示“要剪掉的通道”（以前是 weight[submask] = 0）
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[0],
                                                      dtype=torch.bool,
                                                      device=weight.device)
                                    keep[submask] = False

                                new_out = int(keep.sum().item())
                                new_weight = weight[keep].clone()
                                linear.out_features = new_out
                                linear.weight = torch.nn.Parameter(new_weight)

                                if linear.bias is not None:
                                    new_bias = linear.bias.data[keep].clone()
                                    linear.bias = torch.nn.Parameter(new_bias)

                                # 同时更新这一层的 head 数（可选，但通常需要）
                                attn_block = getattr(layers[i], layer_mapping['attn']['block'])
                                # head 级的 keep：从 submask_layer 推出来
                                if submask_layer.dtype == torch.bool:
                                    head_keep = ~submask_layer
                                else:
                                    head_keep = torch.ones_like(submask_layer,
                                                                dtype=torch.bool,
                                                                device=submask_layer.device)
                                    head_keep[submask_layer] = False
                                new_num_heads = int(head_keep.sum().item())
                                if hasattr(attn_block, "num_heads"):
                                    attn_block.num_heads = new_num_heads
                                if hasattr(attn_block, "num_attention_heads"):
                                    attn_block.num_attention_heads = new_num_heads

                            elif name in [layer_mapping['attn']['k'], layer_mapping['attn']['v']]:
                                # K/V：不改权重的 shape，只用 head 级 mask 交给 hook 屏蔽
                                attn_block = getattr(layers[i], layer_mapping['attn']['block'])
                                attn_block.gqa_mask_record.data[submask] = 0

                            elif name == layer_mapping['attn']['o']:
                                linear = find_layers(layers[i])[name]
                                weight = linear.weight.data  # [out_features, in_features]

                                # 原来的 submask 表示要剪掉的输入通道（以前是 weight[:, submask] = 0）
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[1],
                                                      dtype=torch.bool,
                                                      device=weight.device)
                                    keep[submask] = False

                                new_in = int(keep.sum().item())
                                new_weight = weight[:, keep].clone()
                                linear.in_features = new_in
                                linear.weight = torch.nn.Parameter(new_weight)

            # ================== MLP 部分（统一做结构化剪枝） ==================
            for i in range(len(layers)):
                if f"{i}.{layer_mapping['mlp']['block']}" in mask:
                    submask_layer = mask[f"{i}.{layer_mapping['mlp']['block']}"]

                    if 'g' in layer_mapping['mlp']:
                        submask_u = submask_layer
                        submask_g = submask_layer
                        submask_d = submask_layer

                        for name, vis_name, submask in zip(
                                [layer_mapping['mlp']['u'], layer_mapping['mlp']['g'],
                                 layer_mapping['mlp']['d']],
                                [layer_mapping['mlp']['u_name'],
                                 layer_mapping['mlp']['g_name'],
                                 layer_mapping['mlp']['d_name']],
                                [submask_u, submask_g, submask_d]):

                            linear = find_layers(layers[i])[name]
                            weight = linear.weight.data

                            if name in [layer_mapping['mlp']['d']]:
                                # 原来：[:, submask] = 0  -> 列置零
                                # 现在：按列裁剪 -> in_features 变小
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
                                    keep[submask] = False
                                # ==== 这里做 8 对齐（只能少 mask，一定是往上补）====
                                # keep_num = int(keep.sum().item())
                                # if keep_num % 8 != 0:
                                #     target_keep = min(((keep_num + 7) // 8) * 8, keep.numel())
                                #     need_extra = target_keep - keep_num
                                #     if need_extra > 0:
                                #         cand = torch.nonzero(~keep, as_tuple=False).view(-1)
                                #         if cand.numel() > 0:
                                #             perm = torch.randperm(cand.numel(), device=cand.device)
                                #             restore = cand[perm[:need_extra]]
                                #             keep[restore] = True
                                # ==================================================
                                new_in = int(keep.sum().item())
                                new_weight = weight[:, keep].clone()
                                linear.in_features = new_in
                                linear.weight = torch.nn.Parameter(new_weight)
                            else:
                                # 原来：weight[submask] = 0  -> 行置零
                                # 现在：按行裁剪 -> out_features 变小
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
                                    keep[submask] = False
                                # # ==== 这里做 8 对齐 ====
                                # keep_num = int(keep.sum().item())
                                # if keep_num % 8 != 0:
                                #     target_keep = min(((keep_num + 7) // 8) * 8, keep.numel())
                                #     need_extra = target_keep - keep_num
                                #     if need_extra > 0:
                                #         cand = torch.nonzero(~keep, as_tuple=False).view(-1)
                                #         if cand.numel() > 0:
                                #             perm = torch.randperm(cand.numel(), device=cand.device)
                                #             restore = cand[perm[:need_extra]]
                                #             keep[restore] = True
                                # # ======================
                                new_out = int(keep.sum().item())
                                new_weight = weight[keep].clone()
                                linear.out_features = new_out
                                linear.weight = torch.nn.Parameter(new_weight)
                                if linear.bias is not None:
                                    new_bias = linear.bias.data[keep].clone()
                                    linear.bias = torch.nn.Parameter(new_bias)

                    else:
                        submask_u = submask_layer
                        submask_d = submask_layer

                        for name, vis_name, submask in zip(
                                [layer_mapping['mlp']['u'],
                                 layer_mapping['mlp']['d']],
                                [layer_mapping['mlp']['u_name'],
                                 layer_mapping['mlp']['d_name']],
                                [submask_u, submask_d]):

                            linear = find_layers(layers[i])[name]
                            weight = linear.weight.data

                            if name in [layer_mapping['mlp']['d']]:
                                # down：按列裁剪
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[1], dtype=torch.bool, device=weight.device)
                                    keep[submask] = False
                                # ==== 8 对齐 ====
                                # keep_num = int(keep.sum().item())
                                # if keep_num % 8 != 0:
                                #     target_keep = min(((keep_num + 7) // 8) * 8, keep.numel())
                                #     need_extra = target_keep - keep_num
                                #     if need_extra > 0:
                                #         cand = torch.nonzero(~keep, as_tuple=False).view(-1)
                                #         if cand.numel() > 0:
                                #             perm = torch.randperm(cand.numel(), device=cand.device)
                                #             restore = cand[perm[:need_extra]]
                                #             keep[restore] = True
                                # ================
                                new_in = int(keep.sum().item())
                                new_weight = weight[:, keep].clone()
                                linear.in_features = new_in
                                linear.weight = torch.nn.Parameter(new_weight)
                            else:
                                # up：按行裁剪
                                if submask.dtype == torch.bool:
                                    keep = ~submask
                                else:
                                    keep = torch.ones(weight.shape[0], dtype=torch.bool, device=weight.device)
                                    keep[submask] = False
                                # ==== 8 对齐 ====
                                # keep_num = int(keep.sum().item())
                                # if keep_num % 8 != 0:
                                #     target_keep = min(((keep_num + 7) // 8) * 8, keep.numel())
                                #     need_extra = target_keep - keep_num
                                #     if need_extra > 0:
                                #         cand = torch.nonzero(~keep, as_tuple=False).view(-1)
                                #         if cand.numel() > 0:
                                #             perm = torch.randperm(cand.numel(), device=cand.device)
                                #             restore = cand[perm[:need_extra]]
                                #             keep[restore] = True
                                # ================
                                new_out = int(keep.sum().item())
                                new_weight = weight[keep].clone()
                                linear.out_features = new_out
                                linear.weight = torch.nn.Parameter(new_weight)
                                if linear.bias is not None:
                                    new_bias = linear.bias.data[keep].clone()
                                    linear.bias = torch.nn.Parameter(new_bias)


            self.save_helper.append("actual_mask", mask)
        return checkpoint

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


    def get_imps(self):
        return self.W_metrics

    def finishing_pruning(self, real_pruning=True):
        pass

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
