import os
import re

import numpy as np
import random
import torch
import tqdm
from datasets import load_dataset, DownloadMode, concatenate_datasets, load_from_disk, Dataset, DatasetDict

from modules.data.utils import generate_prompt, tokenize


# Set random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for NumPy and PyTorch for reproducibility.

    Args:
        seed (int): The random seed.
    """
    np.random.seed(seed)
    torch.random.manual_seed(seed)


# Wrapper class for tokenized input IDs
class TokenizerWrapper:
    """
    Wrapper class for tokenized input IDs.

    Args:
        input_ids (tensor): The tokenized input IDs from the tokenizer.
    """

    def __init__(self, input_ids):
        self.input_ids = input_ids


# Load and process PTB (Penn Treebank) dataset
def get_ptb(nsamples, seed, seqlen, tokenizer):
    """
    Load and process PTB (Penn Treebank) dataset.

    Args:
        nsamples (int): Number of samples to extract.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for each sample.
        tokenizer (Tokenizer): Tokenizer to use for text encoding.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test set.
    """
    # Load train and test datasets
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the Wikitext-2 dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded test dataset.
    """
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    # traindata = load_dataset('text', data_files='datasets/wikitext/wiki.train.raw', split="train")
    # testdata = load_dataset('text', data_files='datasets/wikitext/wiki.test.raw', split="train")

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc


# Load and process C4 (Common Crawl) dataset
def get_c4(nsamples, seed, seqlen, tokenizer):
    """
    Load and process the C4 (Common Crawl) dataset.

    Args:
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation dataset.
    """
    # Load train and validation datasets
    traindata = load_dataset('allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train')
    valdata = load_dataset('allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
                           split='validation')
    # traindata = load_dataset('json', data_files={'train': 'datasets/c4/c4-train.00000-of-01024.json.gz'}, split='train')
    # valdata = load_dataset('json', data_files={'validation': 'datasets/c4/c4-validation.00000-of-00008.json.gz'}, split='validation')

    # Generate samples from training set using random seed and specified sequence length
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


def get_pajama(
        nsamples,
        seed,
        seqlen,
        tokenizer,
):
    random.seed(seed)

    raw = load_dataset('DKYoon/SlimPajama-6B')['train']
    proportions = {
        "RedPajamaC4": 0.492,
        "RedPajamaStackExchange": 0.01,
        "RedPajamaCommonCrawl": 0.361 / 3,
        "RedPajamaGithub": 0.008,
        "RedPajamaWikipedia": 0.031,
        "RedPajamaArXiv": 0.007 / 20,
        "RedPajamaBook": 0.091 / 200,
    }
    total_p = sum(proportions.values())
    proportions = {k: v / total_p for k, v in proportions.items()}

    train_splits = {}
    test_splits = []
    for name, prop in proportions.items():
        ds_sub = raw.filter(lambda x: x['meta']['redpajama_set_name'] == name)
        n_test = int(3000 * prop)
        split = ds_sub.train_test_split(test_size=n_test, seed=seed)
        train_splits[name] = split['train']
        test_splits.append(split['test'])

    raw_counts = {name: nsamples * prop for name, prop in proportions.items()}
    floor_counts = {name: int(np.floor(cnt)) for name, cnt in raw_counts.items()}
    remain = nsamples - sum(floor_counts.values())
    remainders = {name: raw_counts[name] - floor_counts[name] for name in proportions}
    for name in sorted(remainders, key=remainders.get, reverse=True)[:remain]:
        floor_counts[name] += 1

    trainloader = []
    for name, n_sub in floor_counts.items():
        ds_sub = train_splits[name]
        for _ in range(n_sub):
            while True:
                idx = random.randint(0, len(ds_sub) - 1)
                text = ds_sub[idx]['text']
                enc = tokenizer(text, return_tensors='pt')
                if enc.input_ids.shape[1] > seqlen:
                    break
            L = enc.input_ids.shape[1]
            start = random.randint(0, L - seqlen - 1)
            chunk = enc.input_ids[:, start:start + seqlen]
            inp = chunk
            tar = chunk.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

    test_union = concatenate_datasets(test_splits)
    texts = test_union[:1100]['text']
    big_str = " ".join(texts)
    valenc = tokenizer(big_str, return_tensors='pt').input_ids
    valenc = valenc[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc


# Function to select the appropriate loader based on dataset name
def get_loaders(name='wikitext2', nsamples=128, seed=0, seqlen=2048, tokenizer=None, data_path=None, base_model=None):
    """
    Select the appropriate loader based on dataset name.

    Args:
        name (str): The name of the dataset ('wikitext2', 'c4', or 'ptb').
        nsamples (int): Number of samples to generate from the training set.
        seed (int): Random seed for reproducibility.
        seqlen (int): Sequence length for generated samples.
        tokenizer (Tokenizer): Tokenizer instance for encoding texts.

    Returns:
        tuple: A tuple containing trainloader (list of input and target pairs) and encoded validation/test set.
    """
    # Determine which dataset to use based on 'name' parameter and return corresponding loader
    total_budget = nsamples * seqlen
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    elif "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer)
    elif "ptb" in name:
        return get_ptb(nsamples, seed, seqlen, tokenizer)
    elif "pajama" in name:
        return get_pajama(nsamples, seed, seqlen, tokenizer)
    elif "boolq_ori" in name:
        return get_boolq(nsamples, seed, seqlen, tokenizer)
    elif "arc_easy" in name:
        return get_arc_easy(total_budget, seed, tokenizer)
    elif "hellaswag_pad" in name:
        return get_hellaswag(nsamples, seed, seqlen, tokenizer)
    else:
        def generate_and_tokenize_prompt(data_point):
            full_prompt = generate_prompt(data_point)
            tokenized_full_prompt = tokenize(full_prompt, tokenizer, seqlen + 1, base_model)
            return tokenized_full_prompt

        if data_path.endswith(".json"):  # todo: support jsonl
            data = load_dataset("json", data_files=data_path)
        else:
            data = load_dataset(data_path)
        train_data = data['train']
        random.seed(seed)
        trainloader = []
        for _ in tqdm.tqdm(range(nsamples)):
            while True:
                i = random.randint(0, len(train_data) - 1)
                trainenc = generate_and_tokenize_prompt(train_data[i])
                for x, y in trainenc.data.items():
                    trainenc[x] = torch.tensor(y).view(1, -1)
                if trainenc.input_ids.shape[1] > seqlen:
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        return trainloader, None


if __name__ == "__main__":
    get_loaders('wikitext2', seed=0, seqlen=2048, tokenizer=None)
