import torch
from torch import nn
from tqdm import tqdm

from modules.eval.setup_eval import eval_ppl
from . import non_uniform_pruner
from .layerwrapper import *
from .utils import *

logger = logging.getLogger(__name__)


class shortGPT_sp(non_uniform_pruner):
    def __init__(self, model, config, data):
        super().__init__(model, config, data)

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name in ['baseline_short_GPT']:
            self.baseline_short_GPT()
        else:
            raise Exception

    def baseline_short_GPT(self):
        self.before_pruning()

        layers = self.get_layers()
        n_samples = self.process_data()

        seq_len = self.config.task.prune.prune_dataset.seq_len
        pruning_ratio = self.config.task.prune.ratio

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(self.get_model(), self.data, n_samples,
                                                                                 seq_len)
        current_cos_sim = self.obtain_information(inps.clone(), attention_mask, position_ids, n_samples)
        order = torch.argsort(current_cos_sim)
        index = 0
        while (True):
            cur_sparsity = self.check_sparsity(real_pruning=True)
            if cur_sparsity >= pruning_ratio:
                print()
                break
            else:
                layers[order[index]] = nn.Identity()
                index += 1

        self.finishing_pruning(real_pruning=True)

    def obtain_information(self, input, attention_mask, position_ids, n_samples):
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
            return outputs

        layers = self.get_layers()
        current_cos_sim = []

        inputs = input.detach().clone()
        for j in tqdm(range(0, len(layers)), desc="Obtaining following layers' cosine similarity"):
            current_layer = layers[j]
            outputs = forward_layer(current_layer, inputs)
            current_cos_sim.append(BI(inputs, outputs).sum().cpu().item())
            inputs, outputs = outputs, inputs

        current_cos_sim = torch.tensor(current_cos_sim)
        return current_cos_sim

    def step(self):
        pass

    def get_imps(self):
        return self.W_metrics

    def process_data(self):
        if not self.data_processed and self.config.task.prune.prune_dataset.type in ['downstream']:
            first_elems = []
            for tup in self.data:
                flat = tup[0].view(-1)  # 扁平化
                nonzero = flat[flat != 0]  # 过滤掉所有值为 0 的元素
                first_elems.append(nonzero)
            big_tensor = torch.cat(first_elems, dim=0)  # shape [L], L = sum_i len(tup_i[0])

            seq_len = self.config.task.prune.prune_dataset.seq_len

            # 3b. 切分成 list
            chunked = big_tensor.split(seq_len)  # 返回 tuple，每项 shape [segment_size]
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
