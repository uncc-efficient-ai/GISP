import copy
import gc

import torch
import numpy as np
from tqdm import tqdm
import random

from datasets import load_dataset, DownloadConfig
from torch.utils.data.dataset import Dataset

wikitext2_traindata = None
wikitext2_testdata = None
ptb_traindata = None
ptb_valdata = None


def PPLMetric(model, tokenizer, datasets, seq_len=128, batch_size=16, device="cuda"):
    metric = {}
    for dataset in datasets:
        _, test_loader = get_loaders(dataset, tokenizer, seq_len=seq_len, batch_size=batch_size)
        ppl = llama_eval(model, test_loader, device)
        metric[dataset] = ppl
        print(metric)
    return metric


@torch.no_grad()
def llama_eval(model, test_lodaer, device):
    nlls = []
    n_samples = 0
    for batch in tqdm(test_lodaer):
        batch = batch.to(device)
        output = model(batch)
        lm_logits = output.logits

        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = batch[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss)
    # print(torch.cat(nlls, dim=-1).mean())
    ppl = np.exp(torch.cat(nlls, dim=-1).mean().item())
    return ppl.item()


def get_wikitext2(seq_len, tokenizer):
    global wikitext2_traindata
    global wikitext2_testdata
    if wikitext2_traindata is not None and wikitext2_testdata is not None:
        return wikitext2_traindata, wikitext2_testdata
    else:
        download_config = DownloadConfig(max_retries=5)
        traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', trust_remote_code=True,
                                 download_config=download_config)

        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', trust_remote_code=True,
                                download_config=download_config)
    wikitext2_traindata = traindata
    wikitext2_testdata = testdata
    return traindata, testdata


def get_ptb(seq_len, tokenizer):
    global ptb_traindata
    global ptb_valdata
    if ptb_traindata is not None and ptb_valdata is not None:
        return ptb_traindata, ptb_valdata
    else:
        download_config = DownloadConfig(max_retries=5)
        traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train', trust_remote_code=True,
                                 download_config=download_config)
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation', trust_remote_code=True,
                               download_config=download_config)
        ptb_traindata = traindata
        ptb_valdata = valdata
    return traindata, valdata


class IndexDataset(Dataset):
    def __init__(self, tensors):
        self.tensors = tensors

    def __getitem__(self, index):
        return self.tensors[index]

    def __len__(self):
        return len(self.tensors)


def process_data(samples, tokenizer, seq_len, field_name):
    test_ids = tokenizer("\n\n".join(samples[field_name]), return_tensors='pt').input_ids[0]
    test_ids_batch = []
    nsamples = test_ids.numel() // seq_len

    for i in range(nsamples):
        batch = test_ids[(i * seq_len):((i + 1) * seq_len)]
        test_ids_batch.append(batch)
    test_ids_batch = torch.stack(test_ids_batch)
    return IndexDataset(tensors=test_ids_batch)


def get_loaders(name, tokenizer, seq_len=2048, batch_size=8):
    if 'wikitext2' in name:
        train_data, test_data = get_wikitext2(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'text')
    if 'ptb' in name:
        train_data, test_data = get_ptb(seq_len, tokenizer)
        test_dataset = process_data(test_data, tokenizer, seq_len, 'sentence')

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_data, test_loader


def eval(model, tokenizer):
    with torch.no_grad():
        # model.to('cpu')
        # modelc = copy.deepcopy(model)
        # modelc.to('cuda')
        # modelc.half()
        # modelc.eval()
        ppl = PPLMetric(model, tokenizer, ['wikitext2', 'ptb'], 128, device=next(model.parameters()).device)
        # result = {"empty": 0}
        # modelc.to('cpu')
    # del modelc
    gc.collect()
    # model.to('cuda')
    return ppl
