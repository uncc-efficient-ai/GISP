# Includes structural compression helpers adapted from FLAP (Apache-2.0).
# Modified for GISP layer mappings, masks, and model architectures.
# See LICENSE and NOTICE in the repository root.

import logging
from collections import defaultdict
import os
import numpy as np
import torch
from torch import nn
from transformers import Conv1D
import torch.nn.functional as F
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class InformationSaveHelper:
    def __init__(self, config, self_logger):
        self.cur_data = {}
        self.config = config
        self.output_folder = config.task.output_folder
        self.self_logger = self_logger
        self.history_saved_data = []

    def clean(self):
        self.cur_data = {}

    def append(self, key, value):
        self.cur_data[key] = value

    def remove(self, key):
        del self.cur_data[key]

    def save(self, current_sparsity, extra_path=None):
        # for key, value in self.cur_data.items():
        #     if isinstance(value, torch.Tensor):
        #         self.cur_data[key] = value.cpu()
        if extra_path is not None:
            save_dir = f'{self.output_folder}/{extra_path}'
            os.makedirs(save_dir, exist_ok=True)
            target_path = f'{self.output_folder}/{extra_path}/sp_{current_sparsity}.pth'
        else:
            target_path = f'{self.output_folder}/sp_{current_sparsity}.pth'
        torch.save(self.cur_data, target_path)
        self.self_logger.info(f"Data bundle saved to {target_path}")
        self.history_saved_data.append((current_sparsity, target_path))
        self.clean()
        return target_path

    def instant_save(self, data, name):
        torch.save(data, f'{self.output_folder}/{name}.pth')
        self.self_logger.info(f"Instant data saved to {self.output_folder}/{name}.pth")
        return f'{self.output_folder}/{name}.pth'

    def delete_sparsity(self, current_sparsity):
        record = [h for h in self.history_saved_data if h[0] == current_sparsity]
        if len(record) != 0:
            file_path = record[0][1]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    self.self_logger.info(f"delete: {file_path} due to rollback")
                except Exception as e:
                    self.self_logger.info(f"delete: '{file_path}' raised exception: {e}")
            self.history_saved_data = [h for h in self.history_saved_data if h[0] != current_sparsity]
        else:
            self.self_logger.info(f"Current sparsity not in history record.")

    def delete(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                self.self_logger.info(f"delete: {checkpoint_path}")
            except Exception as e:
                self.self_logger.info(f"delete: '{checkpoint_path}' raised exception: {e}")
                return False
        return True

    def load(self, checkpoint_path):
        self.self_logger.info(f"Checkpoint data loaded from {checkpoint_path}")
        return torch.load(checkpoint_path)

    def load_sparsity(self, target_sparsity, extra_path=None):
        if extra_path is not None:
            self.self_logger.info(
                f"Checkpoint data loaded from {self.output_folder}/{extra_path}/sp_{target_sparsity}.pth")
            return torch.load(f"{self.output_folder}/{extra_path}/sp_{target_sparsity}.pth")
        else:
            self.self_logger.info(f"Checkpoint data loaded from {self.output_folder}/sp_{target_sparsity}.pth")
            return torch.load(f"{self.output_folder}/sp_{target_sparsity}.pth")

def show(tensor_list, label, title):
    tensor_list = [t.cpu() for t in tensor_list]
    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Tensor Value', color='red')
    ax1.plot(range(len(tensor_list)), tensor_list, marker='o', color='red', label=label)

    plt.title(title)
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{title}.png')
    plt.tight_layout()
    plt.close(fig)


def show_cosine_sim(tensor_list, tensor_list_2, title=None, tensor_list_3=None):
    tensor_list = [t.cpu() for t in tensor_list]
    tensor_list_2 = [t.cpu() for t in tensor_list_2]

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 主 y 轴：tensor 值
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Tensor Value (Cos Sim or angular-dis)', color='blue')
    # ax1.plot(range(len(tensor_list)), tensor_list, marker='o', color='blue', label='Cos Sim')
    #
    # # # 主 y 轴：tensor 值
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Tensor Value (l2 distance)', color='red')
    # ax2.plot(range(len(tensor_list_2)), tensor_list_2, marker='x', color='red', label='l2 distance')

    if tensor_list_3 is not None:
        tensor_list_3 = tensor_list_3.cpu()
        ax1.plot(range(len(tensor_list_3)), tensor_list_3, marker='o', color='green', label='angular-dis normalized')
    # # 找到最大值及其索引
    # max_v = max(tensor_list)
    # max_i = tensor_list.index(max_v)

    # # 次 y 轴：剪枝比率
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Pruning Ratio (%)', color='red')

    # # 计算左侧递增和右侧递减
    # left_line = np.linspace(init_ratio, 100, max_i + 1 - start)
    # right_line = np.linspace(100, 0, len(tensor_list) - max_i - 1)
    #
    # # 画两条线，左边从0%到100%递增，右边从100%到0%递减
    # ax2.plot(range(start, max_i + 1), left_line, marker='o', linestyle='-', color='red',
    #          label='Left pruning ratio (0% to 100%)')
    # ax2.plot(range(max_i, len(tensor_list) - 1), right_line, marker='o', linestyle='-', color='green',
    #          label='Right pruning ratio (100% to 0%)')
    # ax2.tick_params(axis='y', labelcolor='red')

    # 设置 y 轴范围，让左侧 y 轴范围更大
    # ax1.set_ylim(0.5, 1.0)  # 调整左侧 y 轴范围
    # ax2.set_ylim(0, 110)

    # 添加标题和图例
    title = title if title is not None else f'Cos Sim for layers 0 to {len(tensor_list) - 1} in llama2-7b'
    plt.title(title)
    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 显示网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{title}.png')
    plt.tight_layout()
    # plt.show()
    plt.close(fig)
    # return left_line, right_line


def show_diagram_dict(tensor_dict, title, save_location=''):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    for key, values in tensor_dict.items():
        value_cpu = [v.cpu() for v in values]
        ax1.plot(range(len(value_cpu)), value_cpu, marker='o', label=key)  # 使用折线图，并在每个数据点添加圆点标记

    # 设置图表标题和轴标签
    plt.title(title)
    ax1.set_xlabel('Layer index')
    ax1.set_ylabel('Pruning ratio')

    fig.legend(loc='upper right', bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)

    # 显示网格线以便更容易读取值
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_location in ['']:
        plt.savefig(f'{title}.png')
    else:
        plt.savefig(f'{save_location}/{title}.png')
    # 显示图表
    # plt.show()
    plt.close()


def show_diagram_list(tensor_list, title, save_location=''):
    tensor_list = [t.cpu() for t in tensor_list]
    # 创建图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(tensor_list)), tensor_list, marker='o')  # 使用折线图，并在每个数据点添加圆点标记

    # 设置图表标题和轴标签
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Value')

    # 显示网格线以便更容易读取值
    plt.grid(True, linestyle='--', alpha=0.7)
    if save_location in ['']:
        plt.savefig(f'{title}.png')
    else:
        plt.savefig(f'{save_location}/{title}.png')
    # 显示图表
    # plt.show()
    plt.close()


def show_diagram(tensor_list, title):
    if isinstance(tensor_list, dict):
        keys = [int(k) for k in tensor_list.keys()]
        values = [v.cpu().item() for v in tensor_list.values()]  # 将tensor转换为Python数值
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(keys, values, marker='o')  # 使用折线图，并在每个数据点添加圆点标记

        # 设置图表标题和轴标签
        plt.title(title)
        plt.xlabel('Layer index')
        plt.ylabel('Value')

        # 显示网格线以便更容易读取值
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{title}.png')
        # 显示图表
        # plt.show()
    elif isinstance(tensor_list, list):
        # check element
        if isinstance(tensor_list[0], tuple):
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(tensor_list)), [t[0].cpu() for t in tensor_list], marker='o',
                     label='mha')  # 使用折线图，并在每个数据点添加圆点标记
            plt.plot(range(len(tensor_list)), [t[1].cpu() for t in tensor_list], marker='x',
                     label='mlp')  # 使用折线图，并在每个数据点添加圆点标记
            plt.plot(range(len(tensor_list)), [t[2].cpu() for t in tensor_list], marker='s',
                     label='avg')  # 使用折线图，并在每个数据点添加圆点标记

            # 设置图表标题和轴标签
            plt.title(title)
            plt.xlabel('Layer index')
            plt.ylabel('Pruning ratio')

            # 显示网格线以便更容易读取值
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f'{title}.png')
            # 显示图表
            # plt.show()
        else:
            tensor_list = [t.cpu() for t in tensor_list]
            # 创建图表
            plt.figure(figsize=(10, 6))
            plt.plot(range(len(tensor_list)), tensor_list, marker='o')  # 使用折线图，并在每个数据点添加圆点标记

            # 设置图表标题和轴标签
            plt.title(title)
            plt.xlabel('Layer index')
            plt.ylabel('Value')

            # 显示网格线以便更容易读取值
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f'{title}.png')
            # 显示图表
            # plt.show()
    else:
        tensor_list = tensor_list.cpu()
        # 创建图表
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(tensor_list)), tensor_list, marker='o')  # 使用折线图，并在每个数据点添加圆点标记

        # 设置图表标题和轴标签
        plt.title(title)
        plt.xlabel('Layer index')
        plt.ylabel('Value')

        # 显示网格线以便更容易读取值
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(f'{title}.png')
        # 显示图表
        # plt.show()
    plt.close()
    # plt.close()  # 关闭图形，释放内存


def BI(A, B):
    Bsz, S, D = A.shape
    A_flat = A.reshape(-1, D)  # [N, D]
    B_flat = B.reshape(-1, D)  # [N, D]

    sim = torch.nn.functional.cosine_similarity(
        A_flat, B_flat, dim=-1
    ).nan_to_num(nan=0.5)  # [N]
    return 1 - sim


def cosine_similarity(A, B):
    sim = F.cosine_similarity(A, B, dim=-1)
    token_mean = torch.mean(sim, dim=1)
    sample_mean = torch.mean(token_mean, dim=0)
    return sim, token_mean, sample_mean


def dot_product_similarity(A, B):
    # 在最后一个维度（4096）上做点乘
    sim = (A * B).sum(dim=-1)  # shape: (500, 256)

    # 在token维度（256）上求平均
    token_mean = torch.mean(sim, dim=1)  # shape: (500,)

    # 在sample维度（500）上求平均
    sample_mean = torch.mean(token_mean, dim=0)  # shape: scalar

    return sim, token_mean, sample_mean


def angular_distance(A, B):
    sim = cosine_similarity(A, B)[2]
    cos_sim = torch.clamp(sim, -1.0, 1.0)
    distance = torch.acos(cos_sim)
    return distance


def l2_distance(A, B):
    l2_distance = torch.norm(A - B)
    # l2_distance_token_wise = torch.norm(A - B, dim=(1, 2)).mean(dim=0)
    return l2_distance, l2_distance


def std(A, B):
    std_A = torch.std(A)
    std_B = torch.std(B)
    std_d = torch.abs(std_A - std_B)
    std_token_wise_A = torch.std(A, dim=(1, 2)).mean(dim=0)
    std_token_wise_B = torch.std(B, dim=(1, 2)).mean(dim=0)
    std_token_wise = torch.abs(std_token_wise_A - std_token_wise_B)
    return std_d, std_token_wise


def mean(A, B):
    mean_A = torch.mean(A)
    mean_B = torch.mean(B)
    mean_d = torch.abs(mean_A - mean_B)
    mean_token_wise_A = torch.mean(A, dim=(1, 2)).mean(dim=0)
    mean_token_wise_B = torch.mean(B, dim=(1, 2)).mean(dim=0)
    mean_token_wise = torch.abs(mean_token_wise_A - mean_token_wise_B)
    return mean_d, mean_token_wise


def kl_divergence(A, B):
    temperature = 0.1  # 或者其他值
    scaled_A = A / temperature

    input_prob = F.softmax(A, dim=-1)
    output_prob = F.softmax(B, dim=-1)
    kl_div = F.kl_div(output_prob.log(), input_prob, reduction='none')
    kl_div = kl_div.sum(-1)
    kl_div = kl_div.mean(1)
    kl_div = kl_div.mean(0)
    return kl_div


def nested_getattr(obj, attr):
    for attr_part in attr.split('.'):
        obj = getattr(obj, attr_part)
    return obj


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, mapping, bias=True, head_dim=128,
             is_gqa=False):
    """
    Compress a model layer by masking or pruning based on the given masks.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.

    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """

    # Real Pruning
    # Attention Weight Pruning
    if attn_mask is not None:
        # Prune the query, key and value projection weights
        # We reduce the size of the weights based on the attention mask

        q = nested_getattr(layer, mapping['attn']['q'])
        k = nested_getattr(layer, mapping['attn']['k'])
        v = nested_getattr(layer, mapping['attn']['v'])
        o = nested_getattr(layer, mapping['attn']['o'])

        if is_gqa:
            repeat_count = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
            retain_heads_kv = torch.count_nonzero(attn_mask)
            retain_heads_qo = retain_heads_kv * repeat_count
            attn_mask_kv = attn_mask.repeat_interleave(head_dim)
            attn_mask_qo = attn_mask_kv.repeat_interleave(repeat_count)
            for m in [k, v]:
                m.weight.data = m.weight.data[torch.where(attn_mask_kv)[0]]
                m.out_features = attn_mask_kv.sum().item()

                if m.bias is not None:
                    m.bias.data = m.bias.data[torch.where(attn_mask_kv)[0]]
            for m in [q]:
                m.weight.data = m.weight.data[torch.where(attn_mask_qo)[0]]
                m.out_features = attn_mask_qo.sum().item()

                if m.bias is not None:
                    m.bias.data = m.bias.data[torch.where(attn_mask_qo)[0]]
            output_weight = o.weight.data

            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask_qo.to(device)) @ output_weight.T)

            # Prune the output projection weight
            output_weight = o.weight.data[:, torch.where(attn_mask_qo)[0]]
            # Update layer configurations for the new output shape after pruning
            if hasattr(layer, 'attn'):
                layer.attn.num_heads = retain_heads_qo
                layer.attn.num_key_value_heads = retain_heads_kv
                layer.attn.num_key_value_groups = layer.attn.num_heads // layer.attn.num_key_value_heads
                layer.attn.hidden_size = retain_heads_qo * head_dim

            elif hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = retain_heads_qo
                layer.self_attn.num_key_value_heads = retain_heads_kv
                layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                layer.self_attn.hidden_size = retain_heads_qo * head_dim
                if hasattr(layer.self_attn, 'embed_dim'):
                    layer.self_attn.embed_dim = retain_heads_qo * head_dim
            else:
                raise KeyError

            if bias:
                # Re-initialize the Linear layer with new shape and bias
                o.in_features = attn_mask_qo.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                o.bias.data = output_bias

            o.in_features = attn_mask_qo.sum().item()
            # Assign the pruned weights
            o.weight.data = output_weight
        else:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(head_dim)

            for m in [q, k, v]:
                m.weight.data = m.weight.data[torch.where(attn_mask)[0]]
                m.out_features = attn_mask.sum().item()

                if m.bias is not None:
                    m.bias.data = m.bias.data[torch.where(attn_mask)[0]]

            output_weight = o.weight.data

            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)

            # Prune the output projection weight
            output_weight = o.weight.data[:, torch.where(attn_mask)[0]]
            # Update layer configurations for the new output shape after pruning
            if hasattr(layer, 'attn'):
                layer.attn.num_heads = retain_heads
                layer.attn.num_key_value_heads = retain_heads
                layer.attn.num_key_value_groups = layer.attn.num_heads // layer.attn.num_key_value_heads
                layer.attn.hidden_size = retain_heads * head_dim

            elif hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = retain_heads
                layer.self_attn.num_key_value_heads = retain_heads
                layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                layer.self_attn.hidden_size = retain_heads * head_dim
                if hasattr(layer.self_attn, 'embed_dim'):
                    layer.self_attn.embed_dim = retain_heads * head_dim
            else:
                raise KeyError

            if bias:
                # Re-initialize the Linear layer with new shape and bias
                o.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                o.bias.data = output_bias

            o.in_features = attn_mask.sum().item()
            # Assign the pruned weights
            o.weight.data = output_weight

    # MLP Weight Pruning
    if mlp_mask is not None:
        # Prune the up and gate projection weights
        u = nested_getattr(layer, mapping['mlp']['u'])
        d = nested_getattr(layer, mapping['mlp']['d'])
        if 'g' in mapping['mlp']:
            g = nested_getattr(layer, mapping['mlp']['g'])
        else:
            g = None

        for m in [u, g]:
            if m is not None:
                m.weight.data = m.weight.data[torch.where(mlp_mask)[0]]
                m.out_features = mlp_mask.sum().item()
                if m.bias is not None:
                    m.bias.data = m.bias.data[torch.where(mlp_mask)[0]]

        output_weight = d.weight.data

        if mapping['mlp']['block'] != '':
            layer.mlp.intermediate_size = mlp_mask.sum().item()
        if bias:
            # Add the additional bias to compensate for the loss
            output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

        # Prune the down projection weight
        output_weight = d.weight.data[:, torch.where(mlp_mask)[0]]

        if bias:
            # Re-initialize the Linear layer with new shape and bias
            d.in_features = mlp_mask.sum().item()
            # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
            d.bias.data = output_bias
        d.in_features = mlp_mask.sum().item()
        # Assign the pruned weights
        d.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()


def compress_residue(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, mapping, bias=True, head_dim=128,
                     is_gqa=False):
    """
    Compress a model layer by masking or pruning based on the given masks,
    and return the pruned parameters as separate nn.Linear layers.

    Attention: for pruned attention parameters, the pruned parts of q, k, v, o
    will be returned as separate Linear layers whose weight shape corresponds to
    the parts that were removed by the mask.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        head_dim (int, optional): The hidden dimension per head. Defaults to 128.
        is_gqa (bool, optional): Whether using GQA (Grouped Query Attention). Defaults to False.

    Returns:
        dict: A dictionary of new nn.Linear layers representing the pruned parameters.
    """
    pruned_layers = {}  # 用于存储各个剪枝掉的参数对应的 Linear 层

    # ----- Attention Weight Pruning -----
    if attn_mask is not None:
        q = nested_getattr(layer, mapping['attn']['q'])
        k = nested_getattr(layer, mapping['attn']['k'])
        v = nested_getattr(layer, mapping['attn']['v'])
        o = nested_getattr(layer, mapping['attn']['o'])

        if is_gqa:
            repeat_count = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
            retain_heads_kv = int(torch.count_nonzero(attn_mask).item())
            retain_heads_qo = retain_heads_kv * repeat_count

            attn_mask_kv = attn_mask.repeat_interleave(head_dim)
            attn_mask_qo = attn_mask_kv.repeat_interleave(repeat_count)

            # 对 k, v 进行剪枝（行剪枝，即丢弃部分输出维度）
            for m, name in zip([k, v], ['k', 'v']):
                keep_idx = torch.where(attn_mask_kv)[0]
                prune_idx = torch.where(~attn_mask_kv)[0]
                # 保存被剪枝的部分：权重 shape = (num_pruned, in_features)
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                # 构造新的 Linear 层来存储被剪枝参数
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                # 更新原层：只保留 keep_idx 部分
                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(attn_mask_kv.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

            # 对 q 进行剪枝（行剪枝）
            for m, name in zip([q], ['q']):
                keep_idx = torch.where(attn_mask_qo)[0]
                prune_idx = torch.where(~attn_mask_qo)[0]
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(attn_mask_qo.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

            # 对 o 进行剪枝：注意 o 的剪枝方向在于输入维度（列剪枝）
            if bias:
                output_bias = ((attn_mean_inp * ~attn_mask_qo.to(device)) @ o.weight.data.T)
            keep_idx = torch.where(attn_mask_qo)[0]
            prune_idx = torch.where(~attn_mask_qo)[0]
            pruned_weight = o.weight.data[:, prune_idx].clone().detach()
            # 构造一个新的 Linear 层，其 in_features 为剪枝掉的列数，out_features 保持 o.out_features
            pruned_layer = nn.Linear(len(prune_idx), o.out_features, bias=False)
            pruned_layer.weight.data = pruned_weight
            pruned_layers['o_pruned'] = pruned_layer

            # 更新原 o 层：只保留 keep_idx 列
            output_weight = o.weight.data[:, keep_idx]
            o.in_features = int(attn_mask_qo.sum().item())
            if bias:
                o.bias.data = output_bias
            o.weight.data = output_weight

            # 更新 layer 中 attention 配置
            if hasattr(layer, 'attn'):
                layer.attn.num_heads = retain_heads_qo
                layer.attn.num_key_value_heads = retain_heads_kv
                layer.attn.num_key_value_groups = layer.attn.num_heads // layer.attn.num_key_value_heads
                layer.attn.hidden_size = retain_heads_qo * head_dim
            elif hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = retain_heads_qo
                layer.self_attn.num_key_value_heads = retain_heads_kv
                layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                layer.self_attn.hidden_size = retain_heads_qo * head_dim
                if hasattr(layer.self_attn, 'embed_dim'):
                    layer.self_attn.embed_dim = retain_heads_qo * head_dim
            else:
                raise KeyError("Layer does not have attn or self_attn attribute")

        else:
            # 非 GQA 分支：直接对 q, k, v 使用相同的 mask 处理
            retain_heads = int(torch.count_nonzero(attn_mask).item())
            attn_mask_rep = attn_mask.repeat_interleave(head_dim)

            for m, name in zip([q, k, v], ['q', 'k', 'v']):
                keep_idx = torch.where(attn_mask_rep)[0]
                prune_idx = torch.where(~attn_mask_rep)[0]
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(attn_mask_rep.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

            # 对 o 进行列剪枝
            if bias:
                output_bias = ((attn_mean_inp * ~attn_mask_rep.to(device)) @ o.weight.data.T)
            keep_idx = torch.where(attn_mask_rep)[0]
            prune_idx = torch.where(~attn_mask_rep)[0]
            pruned_weight = o.weight.data[:, prune_idx].clone().detach()
            pruned_layer = nn.Linear(len(prune_idx), o.out_features, bias=False)
            pruned_layer.weight.data = pruned_weight
            pruned_layers['o_pruned'] = pruned_layer

            output_weight = o.weight.data[:, keep_idx]
            o.in_features = int(attn_mask_rep.sum().item())
            if bias:
                o.bias.data = output_bias
            o.weight.data = output_weight

            if hasattr(layer, 'attn'):
                layer.attn.num_heads = retain_heads
                layer.attn.num_key_value_heads = retain_heads
                layer.attn.num_key_value_groups = layer.attn.num_heads // layer.attn.num_key_value_heads
                layer.attn.hidden_size = retain_heads * head_dim
            elif hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = retain_heads
                layer.self_attn.num_key_value_heads = retain_heads
                layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                layer.self_attn.hidden_size = retain_heads * head_dim
                if hasattr(layer.self_attn, 'embed_dim'):
                    layer.self_attn.embed_dim = retain_heads * head_dim
            else:
                raise KeyError("Layer does not have attn or self_attn attribute")

    # ----- MLP Weight Pruning -----
    if mlp_mask is not None:
        u = nested_getattr(layer, mapping['mlp']['u'])
        d = nested_getattr(layer, mapping['mlp']['d'])
        if 'g' in mapping['mlp']:
            g = nested_getattr(layer, mapping['mlp']['g'])
        else:
            g = None

        # 对 u 和 g 进行行剪枝
        for m, name in zip([u, g], ['u', 'g']):
            if m is not None:
                keep_idx = torch.where(mlp_mask)[0]
                prune_idx = torch.where(~mlp_mask)[0]
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(mlp_mask.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

        # 对 d 进行列剪枝
        if bias:
            output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ d.weight.data.T)
        keep_idx = torch.where(mlp_mask)[0]
        prune_idx = torch.where(~mlp_mask)[0]
        pruned_weight = d.weight.data[:, prune_idx].clone().detach()
        pruned_layer = nn.Linear(len(prune_idx), d.out_features, bias=False)
        pruned_layer.weight.data = pruned_weight
        pruned_layers['d_pruned'] = pruned_layer

        output_weight = d.weight.data[:, keep_idx]
        if mapping['mlp']['block'] != '':
            layer.mlp.intermediate_size = int(mlp_mask.sum().item())
        d.in_features = int(mlp_mask.sum().item())
        if bias:
            d.bias.data = output_bias
        d.weight.data = output_weight

    torch.cuda.empty_cache()
    return pruned_layers


def compress_residue_swift(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, mapping, bias=True,
                           head_dim=128,
                           is_gqa=False):
    """
    Compress a model layer by masking or pruning based on the given masks,
    and return the pruned parameters as separate nn.Linear layers.

    Attention: for pruned attention parameters, the pruned parts of q, k, v, o
    will be returned as separate Linear layers whose weight shape corresponds to
    the parts that were removed by the mask.

    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights. True marks retain.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights. True marks retain.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        head_dim (int, optional): The hidden dimension per head. Defaults to 128.
        is_gqa (bool, optional): Whether using GQA (Grouped Query Attention). Defaults to False.

    Returns:
        dict: A dictionary of new nn.Linear layers representing the pruned parameters.
    """
    pruned_layers = {}  # 用于存储各个剪枝掉的参数对应的 Linear 层

    # ----- Attention Weight Pruning -----
    if attn_mask is not None:
        q = nested_getattr(layer, mapping['attn']['q'])
        k = nested_getattr(layer, mapping['attn']['k'])
        v = nested_getattr(layer, mapping['attn']['v'])
        o = nested_getattr(layer, mapping['attn']['o'])

        if is_gqa:
            repeat_count = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
            # attn_mask_kv = attn_mask.repeat_interleave(head_dim)
            attn_mask_qo = attn_mask.repeat_interleave(head_dim)
            layer.kv_binary_mask = attn_mask.clone()

            # 对 q 层继续执行剪枝（使用 attn_mask_qo）
            for m, name in zip([q], ['q']):
                keep_idx = torch.where(attn_mask_qo)[0]
                prune_idx = torch.where(~attn_mask_qo)[0]
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(attn_mask_qo.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

            # 对 o 层执行剪枝（列剪枝，使用 attn_mask_qo）
            if bias:
                output_bias = ((attn_mean_inp * ~attn_mask_qo.to(device)) @ o.weight.data.T)
            keep_idx = torch.where(attn_mask_qo)[0]
            prune_idx = torch.where(~attn_mask_qo)[0]
            pruned_weight = o.weight.data[:, prune_idx].clone().detach()
            pruned_layer = nn.Linear(len(prune_idx), o.out_features, bias=False)
            pruned_layer.weight.data = pruned_weight
            pruned_layers['o_pruned'] = pruned_layer

            output_weight = o.weight.data[:, keep_idx]
            o.in_features = int(attn_mask_qo.sum().item())
            if bias:
                o.bias.data = output_bias
            o.weight.data = output_weight

            # 更新当前层中的 attention 配置
            if hasattr(layer, 'attn'):
                layer.attn.num_heads = int(attn_mask.sum().item())
                # layer.attn.num_key_value_heads = int(attn_mask.sum().item())
                # layer.attn.num_key_value_groups = layer.attn.num_heads // layer.attn.num_key_value_heads
                layer.attn.hidden_size = int(attn_mask_qo.sum().item())
            elif hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = int(attn_mask.sum().item())
                # layer.self_attn.num_key_value_heads = int(attn_mask.sum().item())
                # layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                layer.self_attn.hidden_size = int(attn_mask_qo.sum().item())
                if hasattr(layer.self_attn, 'embed_dim'):
                    layer.self_attn.embed_dim = int(attn_mask_qo.sum().item())
            else:
                raise KeyError("Layer does not have attn or self_attn attribute")

        else:
            # 非 GQA 分支：直接对 q, k, v 使用相同的 mask 处理
            retain_heads = int(torch.count_nonzero(attn_mask).item())
            attn_mask_rep = attn_mask.repeat_interleave(head_dim)

            for m, name in zip([q, k, v], ['q', 'k', 'v']):
                keep_idx = torch.where(attn_mask_rep)[0]
                prune_idx = torch.where(~attn_mask_rep)[0]
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(attn_mask_rep.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

            # 对 o 进行列剪枝
            if bias:
                output_bias = ((attn_mean_inp * ~attn_mask_rep.to(device)) @ o.weight.data.T)
            keep_idx = torch.where(attn_mask_rep)[0]
            prune_idx = torch.where(~attn_mask_rep)[0]
            pruned_weight = o.weight.data[:, prune_idx].clone().detach()
            pruned_layer = nn.Linear(len(prune_idx), o.out_features, bias=False)
            pruned_layer.weight.data = pruned_weight
            pruned_layers['o_pruned'] = pruned_layer

            output_weight = o.weight.data[:, keep_idx]
            o.in_features = int(attn_mask_rep.sum().item())
            if bias:
                o.bias.data = output_bias
            o.weight.data = output_weight

            if hasattr(layer, 'attn'):
                layer.attn.num_heads = retain_heads
                layer.attn.num_key_value_heads = retain_heads
                layer.attn.num_key_value_groups = layer.attn.num_heads // layer.attn.num_key_value_heads
                layer.attn.hidden_size = retain_heads * head_dim
            elif hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = retain_heads
                layer.self_attn.num_key_value_heads = retain_heads
                layer.self_attn.num_key_value_groups = layer.self_attn.num_heads // layer.self_attn.num_key_value_heads
                layer.self_attn.hidden_size = retain_heads * head_dim
                if hasattr(layer.self_attn, 'embed_dim'):
                    layer.self_attn.embed_dim = retain_heads * head_dim
            else:
                raise KeyError("Layer does not have attn or self_attn attribute")

    # ----- MLP Weight Pruning -----
    if mlp_mask is not None:
        u = nested_getattr(layer, mapping['mlp']['u'])
        d = nested_getattr(layer, mapping['mlp']['d'])
        if 'g' in mapping['mlp']:
            g = nested_getattr(layer, mapping['mlp']['g'])
        else:
            g = None

        # 对 u 和 g 进行行剪枝
        for m, name in zip([u, g], ['u', 'g']):
            if m is not None:
                keep_idx = torch.where(mlp_mask)[0]
                prune_idx = torch.where(~mlp_mask)[0]
                pruned_weight = m.weight.data[prune_idx].clone().detach()
                if m.bias is not None:
                    pruned_bias = m.bias.data[prune_idx].clone().detach()
                else:
                    pruned_bias = None
                pruned_layer = nn.Linear(m.in_features, len(prune_idx), bias=(m.bias is not None))
                pruned_layer.weight.data = pruned_weight
                if m.bias is not None:
                    pruned_layer.bias.data = pruned_bias
                pruned_layers[name + '_pruned'] = pruned_layer

                m.weight.data = m.weight.data[keep_idx]
                m.out_features = int(mlp_mask.sum().item())
                if m.bias is not None:
                    m.bias.data = m.bias.data[keep_idx]

        # 对 d 进行列剪枝
        if bias:
            output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ d.weight.data.T)
        keep_idx = torch.where(mlp_mask)[0]
        prune_idx = torch.where(~mlp_mask)[0]
        pruned_weight = d.weight.data[:, prune_idx].clone().detach()
        pruned_layer = nn.Linear(len(prune_idx), d.out_features, bias=False)
        pruned_layer.weight.data = pruned_weight
        pruned_layers['d_pruned'] = pruned_layer

        output_weight = d.weight.data[:, keep_idx]
        if mapping['mlp']['block'] != '':
            layer.mlp.intermediate_size = int(mlp_mask.sum().item())
        d.in_features = int(mlp_mask.sum().item())
        if bias:
            d.bias.data = output_bias
        d.weight.data = output_weight

    torch.cuda.empty_cache()
    return pruned_layers


def prepare_calibration_input(model, dataloader, nsamples, seqlen):
    """
    Prepare inputs for model calibration.

    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded.

    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False
    try:
        layers = model.model.layers
    except AttributeError:
        try:
            layers = model.base_model.layers
        except AttributeError:
            layers = model.base_model.decoder.layers
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]
    else:
        device = model.device
    dtype = next(iter(model.parameters())).dtype
    # dtype = torch.float
    inps = torch.zeros((nsamples, seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            if hasattr(module, "self_attn"):
                self.self_attn = module.self_attn
            elif hasattr(module, "attn"):
                self.attn = module.attn

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            # model(torch.rand_like(batch[0].to(device)))
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids


def find_layers(module, layers=[nn.Linear, Conv1D], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def check_unstr_sparsity(layers):
    logger.info("*" * 30)
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
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


def EMA(data, alpha):
    ema_values = torch.zeros_like(data)
    ema_values[0] = data[0]

    for i in range(1, len(data)):
        ema_values[i] = alpha * data[i] + (1 - alpha) * ema_values[i - 1]
    return ema_values


def pack_data_blocks(trainloader):
    """
    trainloader: List of (inp, labels) tuples, 每个 inp.shape=[1, L_i]
    返回:
      {
        L1: (inps_tensor, labels_tensor),
        L2: (inps_tensor, labels_tensor),
        ...
      }
    其中每个 inps_tensor.shape = [N_i, L_i]
    """
    # 1) 分桶
    buckets = defaultdict(list)
    for inp, labels in trainloader:
        L = inp.size(1)
        buckets[L].append((inp, labels))

    # 2) 每个桶内部拼成大张量
    packed = {}
    for L, samples in buckets.items():
        # 列表拆分，dim=0 拼 batch
        inps = torch.cat([s[0] for s in samples], dim=0)  # [N_i, L]
        labs = torch.cat([s[1] for s in samples], dim=0)  # [N_i, L]
        packed[L] = (inps, labs)

    return packed
