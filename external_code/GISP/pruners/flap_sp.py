# Adapted from FLAP (Apache-2.0).
# Modified for GISP configuration, model handling, and pruning records.
# See LICENSE and THIRD_PARTY_NOTICES.md in the repository root.

from torch import nn
from tqdm import tqdm

from modules.eval.setup_eval import eval_ppl
from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *

logger = logging.getLogger(__name__)


class flap_sp(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['baseline_prune_flap']:
            self.baseline_prune_flap()
        else:
            raise Exception

    def baseline_prune_flap(self):
        self.before_pruning()

        layers = self.get_layers()
        n_samples = self.process_data()

        seq_len = self.config.task.prune.prune_dataset.seq_len
        pruning_ratio = self.config.task.prune.ratio

        structure = self.config.task.prune.structure
        metrics = self.config.task.prune.metrics
        remove_heads = self.config.task.prune.remove_heads
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads
        bias = self.config.task.prune.bias

        if bias:
            for i in range(len(layers)):
                self.model.model.layers[i].self_attn.o_proj.bias = torch.nn.Parameter(
                    torch.zeros_like(self.model.model.layers[i].self_attn.o_proj.bias, device=self.model.device))
                self.model.model.layers[i].mlp.down_proj.bias = torch.nn.Parameter(
                    torch.zeros_like(self.model.model.layers[i].mlp.down_proj.bias, device=self.model.device))
                torch.nn.init.zeros_(self.model.model.layers[i].self_attn.o_proj.bias)
                torch.nn.init.zeros_(self.model.model.layers[i].mlp.down_proj.bias)

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(self.model, self.data, n_samples,
                                                                                 seq_len)

        attn_metric_list, mlp_metric_list = [], []
        attn_baseline_inp_list, mlp_baseline_inp_list = [], []
        attn_mask, mlp_mask = [], []

        """
            'IFV': Input Feature Variance
            'WIFV': Weighted Input Feature Variance
            'WIFN': Weighted Input Feature Norm
        """
        metrics_dict = {
            'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
            'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(
                subset[name].weight.data.pow(2), dim=0),
            'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_inp.reshape((1, -1)))).mean(axis=0),
        }

        def cal_remove_neuron(model, structure, pruning_ratio, remove_heads):
            intermediate_size = model.config.intermediate_size
            hidden_size = model.config.hidden_size
            num_layers = model.config.num_hidden_layers
            if structure == "UL-MM":
                remove_params = pruning_ratio * (
                        intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
                remove_head_params = hidden_size * 4 * (remove_heads // num_layers) * 128
                return int((remove_params - remove_head_params) / (hidden_size * 3))
            else:
                remove_params = num_layers * pruning_ratio * (
                        intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
                remove_head_params = hidden_size * 4 * remove_heads * 128
                return int((remove_params - remove_head_params) / (hidden_size * 3))

        # Split into sub-problems, separate statistics for each module
        for i in tqdm(range(len(layers)), desc="Processing layers"):
            layer = layers[i]
            subset = {}
            subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
            subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map',
                                              {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
                dev = self.model.hf_device_map[f"model.layers.{i}"]
                inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                    dev), position_ids.to(dev)

            wrapped_layers = {}
            for name in subset:
                wrapped_layers[name] = BiasGPT(subset[name], metrics)

            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(n_samples):
                with torch.no_grad():
                    outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

            for h in handles:
                h.remove()

            for name in subset:
                if name == 'self_attn.o_proj':
                    W_metric = metrics_dict[metrics](wrapped_layers, subset, name)
                    if structure == "UL-UM":
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                        thresh = torch.sort(W_metric.cuda())[0][
                            int(pruning_ratio * layer.self_attn.num_heads)].cpu()
                        W_mask = (W_metric >= thresh)
                        attn_mask.append(W_mask)
                    elif structure == "UL-MM":
                        W_metric = W_metric.reshape(-1, 128).sum(dim=1)
                        thresh = torch.sort(W_metric.cuda())[0][
                            remove_heads // len(layers)].cpu()
                        W_mask = (W_metric >= thresh)
                        attn_mask.append(W_mask)
                    else:
                        attn_metric_list.append(W_metric.cpu())
                    attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                else:
                    W_metric = metrics_dict[metrics](wrapped_layers, subset, name)
                    if structure == "UL-UM":
                        thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * pruning_ratio)].cpu()
                        W_mask = (W_metric >= thresh)
                        mlp_mask.append(W_mask)
                    elif structure == "UL-MM":
                        thresh = torch.sort(W_metric.cuda())[0][
                            cal_remove_neuron(self.model, structure, pruning_ratio, remove_heads)].cpu()
                        W_mask = (W_metric >= thresh)
                        mlp_mask.append(W_mask)
                    else:
                        mlp_metric_list.append(W_metric.cpu())
                    mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
                wrapped_layers[name].free()

            inps, outs = outs, inps  # Use the original output as input to the next layer
            torch.cuda.empty_cache()

        standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)

        if structure in ["AL-MM", "AL-AM"]:
            attn_metric = torch.stack(attn_metric_list)
            attn_metric = standarlization(attn_metric)
            attn_metric = attn_metric.reshape(len(layers), -1, 128).mean(dim=2)

            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)

            if structure == "AL-MM":
                sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
                attn_thres = sorted_attn[-int(remove_heads)]
                attn_mask = (attn_metric > attn_thres)  # 1 means retain

                sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
                mlp_thres = sorted_mlp[-cal_remove_neuron(self.model, structure, pruning_ratio, remove_heads)]
                mlp_mask = (mlp_metric > mlp_thres)
            else:
                prune_metric = torch.cat([attn_metric.view(-1), mlp_metric.view(-1)])
                sorted_prune, indices = torch.sort(prune_metric, descending=True)
                compression_weight = torch.ones_like(indices)
                compression_weight[indices < attn_metric.numel()] = 512.0 / 3
                threshold = sorted_prune[torch.argmin(torch.abs(
                    torch.cumsum(compression_weight, 0) - torch.sum(compression_weight) * (1 - pruning_ratio)))]
                attn_mask = (attn_metric > threshold)
                mlp_mask = (mlp_metric > threshold)
        else:
            attn_mask = torch.stack(attn_mask)
            mlp_mask = torch.stack(mlp_mask)

        for idx in range(len(layers)):
            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
                compress(self.model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
                         self.model.hf_device_map[f"model.layers.{idx}"], mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)
            else:
                compress(self.model.model.layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
                         self.model.device, mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)

            if f"model.layers.{i}" in getattr(self.model, 'hf_device_map', {}):
                compress(self.model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
                         self.model.hf_device_map[f"model.layers.{idx}"], mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)
            else:
                compress(self.model.model.layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
                         self.model.device, mapping=self.layer_mapping, head_dim=head_dim,
                         bias=bias)

        self.finishing_pruning(real_pruning=True)

    def step(self):
        pass

    def get_imps(self):
        return self.W_metrics

    def process_data(self):
        if not self.data_processed and self.config.task.prune.prune_dataset.type in ['downstream']:
            first_elems = []
            for tup in self.data:
                flat = tup[0].view(-1)
                nonzero = flat[flat != 0]
                first_elems.append(nonzero)
            big_tensor = torch.cat(first_elems, dim=0)  # shape [L], L = sum_i len(tup_i[0])

            seq_len = self.config.task.prune.prune_dataset.seq_len

            chunked = big_tensor.split(seq_len)
            new_data = []
            for i, chunk in enumerate(chunked):
                if chunk.numel() < seq_len:
                    chunk = F.pad(chunk, (0, seq_len - chunk.numel()), value=0)
                new_data.append((chunk.unsqueeze(0),))
            self.data = new_data
            self.data_processed = True
        return len(self.data)


def count_params(layers):
    layer_params = 0
    for l in layers:
        layer_params += sum(p.numel() for p in l.parameters())
    return layer_params
