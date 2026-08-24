import copy
import os.path
from pathlib import Path

import torch
from datasets import load_dataset
import json
import random
import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)

def get_c4(samples, cutoff_len, tokenizer):
    Path(f"dataset/c4_{samples}_{cutoff_len}").mkdir(exist_ok=True)
    file_path = f"dataset/c4_{samples}_{cutoff_len}/train.json"
    if os.path.exists(cutoff_len):
        dataset = load_dataset("json", data_files=file_path)
        if len(dataset) == samples:
            print(f"load c4 from pre-built {file_path}")
            return dataset

    dataset = load_dataset('allenai/c4',
                           data_files={'train': ['en/c4-train.00000-of-01024.json.gz',
                                                 'en/c4-train.00001-of-01024.json.gz',
                                                 'en/c4-train.00002-of-01024.json.gz',
                                                 'en/c4-train.00003-of-01024.json.gz',
                                                 'en/c4-train.00004-of-01024.json.gz',
                                                 'en/c4-train.00005-of-01024.json.gz',
                                                 'en/c4-train.00006-of-01024.json.gz',
                                                 'en/c4-train.00007-of-01024.json.gz',
                                                 'en/c4-train.00008-of-01024.json.gz',
                                                 'en/c4-train.00009-of-01024.json.gz',
                                                 'en/c4-train.00010-of-01024.json.gz',
                                                 'en/c4-train.00011-of-01024.json.gz',
                                                 'en/c4-train.00012-of-01024.json.gz',
                                                 'en/c4-train.00013-of-01024.json.gz',
                                                 'en/c4-train.00014-of-01024.json.gz',
                                                 'en/c4-train.00015-of-01024.json.gz',
                                                 'en/c4-train.00016-of-01024.json.gz',
                                                 'en/c4-train.00017-of-01024.json.gz',
                                                 'en/c4-train.00018-of-01024.json.gz',
                                                 'en/c4-train.00019-of-01024.json.gz',
                                                 'en/c4-train.00020-of-01024.json.gz',
                                                 'en/c4-train.00021-of-01024.json.gz',
                                                 'en/c4-train.00022-of-01024.json.gz',
                                                 'en/c4-train.00023-of-01024.json.gz',
                                                 'en/c4-train.00024-of-01024.json.gz',
                                                 'en/c4-train.00025-of-01024.json.gz',
                                                 'en/c4-train.00026-of-01024.json.gz',
                                                 'en/c4-train.00027-of-01024.json.gz',
                                                 'en/c4-train.00028-of-01024.json.gz',
                                                 'en/c4-train.00029-of-01024.json.gz',
                                                 ]},
                           split='train')
    print(f"Sampling {samples} data from c4")
    subdata, history = [], []
    for _ in tqdm.tqdm(range(samples)):
        while True:
            i = random.randint(0, len(dataset) - 1)
            trainenc = tokenizer(dataset[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > cutoff_len and i not in history:
                history.append(i)
                break
        subdata.append({"inputs": dataset[i]['text']})
    with open(file_path, 'w') as f:
        f.writelines(json.dumps(subdata))
    return load_dataset("json", data_files=file_path)


def get_e2e(tokenizer: PreTrainedTokenizer, file_path, block_size):
    dataset = LineByLineData2TextTextDataset(tokenizer=tokenizer, file_path=file_path,
                                             block_size=tokenizer.model_max_length if block_size < 0 else min(
                                                 block_size, tokenizer.model_max_length), bos_tok=tokenizer.bos_token,
                                             eos_tok=tokenizer.eos_token, lowdata_token=None)

    return dataset


get_dataset = {'c4': get_c4, 'e2e': get_e2e}


# e2e
class LineByLineData2TextTextDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, file_path: str, block_size: int, bos_tok: str, eos_tok: str,
                 lowdata_token: str):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        # Here, we do not cache the features, operating under the assumption
        # that we will soon use fast multithreaded tokenizers from the
        # `tokenizers` repo everywhere =)
        logger.info("Creating features from dataset file at %s", file_path)

        with open(file_path, encoding="utf-8") as f:
            lines = [line.split('||') for line in f.read().splitlines() if (len(line) > 0 and not line.isspace()
                                                                            and len(line.split('||')) == 2)]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)

        if lowdata_token is None:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} '.format(src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)
        else:
            edited_sents = []
            for src, tgt in zip(src_lines, tgt_lines):
                sent = ' {} {} {} '.format(lowdata_token, src, bos_tok) + tgt + ' {}'.format(eos_tok)
                edited_sents.append(sent)

        batch_encoding = tokenizer(edited_sents, add_special_tokens=True, truncation=True, max_length=block_size,
                                   is_split_into_words=False)
        self.examples = batch_encoding["input_ids"]

        self.labels = copy.deepcopy(self.examples)

        # split into category words:
        ssl_lst = []
        for ss in src_lines:
            ssl = [la.split(':')[0].strip() for la in ss.split('|')]
            # print(ssl)
            ssl_lst.append(ssl)

        self.src_cat = tokenizer(ssl_lst, add_special_tokens=True, truncation=True, max_length=block_size,
                                 is_split_into_words=True)['input_ids']

        self.src_sent = []
        self.tgt_sent = []

        temp_src_len = 0
        temp_tgt_len = 0
        temp_count = 0
        if True:
            separator = tokenizer(bos_tok, add_special_tokens=False)['input_ids'][0]
            for i, elem in enumerate(self.labels):
                sep_idx = elem.index(separator) + 1
                self.src_sent.append(self.examples[i][:sep_idx - 1])
                self.tgt_sent.append(self.examples[i][sep_idx - 1:])
                self.labels[i][:sep_idx] = [-100] * sep_idx
                temp_src_len += sep_idx - 1
                temp_tgt_len += len(elem) - (sep_idx - 1)
                temp_count += 1

        print('tgt_avg: ', temp_tgt_len / temp_count)
        print('src_avg: ', temp_src_len / temp_count)
        print('ratios: ', temp_src_len / temp_tgt_len)

        print(self.labels[0])
        print(self.examples[0])
        print(edited_sents[0])
        print(self.src_sent[0])
        print(self.tgt_sent[0])
        print(self.src_cat[0])
        assert len(self.src_cat) == len(self.examples)

    def __len__(self):
        return len(self.examples)

    # def __getitem__(self, i) -> torch.Tensor:
    def __getitem__(self, i):
        return (torch.tensor(self.examples[i], dtype=torch.long),
                torch.tensor(self.labels[i], dtype=torch.long),
                torch.tensor(self.src_sent[i], dtype=torch.long),
                torch.tensor(self.tgt_sent[i], dtype=torch.long),
                torch.tensor(self.src_cat[i], dtype=torch.long),

                )

