import os
import pickle
from functools import partial
from pathlib import Path

from datasets import load_dataset

from . import data_prune_ds, data_prune
from .data import get_dataset

from .utils import generate_and_tokenize_prompt
from modules.config.config import get_repository_paths


def _cache_file(filename):
    cache_root = get_repository_paths()['cache_root']
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / filename


def _read_cache_file(filename):
    """Prefer the standalone cache root, with a legacy CWD fallback."""
    candidates = (_cache_file(filename), Path.cwd() / filename)
    return next((path for path in candidates if path.exists()), candidates[0])


def make_train_data(tune_config, model_config, seed, tokenizer):
    if tune_config.tune_dataset.pickle.pickle_dump:
        train_cache = _read_cache_file(f'{tune_config.tune_dataset.name}_train_data.pkl')
        val_cache = _read_cache_file(f'{tune_config.tune_dataset.name}_val_data.pkl')
        if train_cache.exists():
            with train_cache.open('rb') as f:
                train_data = pickle.load(f)
        else:
            train_data = None
        if val_cache.exists():
            with val_cache.open('rb') as f:
                val_data = pickle.load(f)
        else:
            val_data = None
    else:
        if tune_config.tune_dataset.mapping:
            if tune_config.tune_dataset.path and tune_config.tune_dataset.path.endswith(".json"):  # todo: support jsonl
                data = load_dataset("json", data_files=tune_config.tune_dataset.path)
            else:
                data = get_dataset[tune_config.tune_dataset.name](tune_config.nsamples, tune_config.cutoff_len,
                                                                  tokenizer)
            if tune_config.val_set_size > 0:
                train_val = data["train"].train_test_split(
                    test_size=tune_config.val_set_size, shuffle=True, seed=seed
                )
                train_data = (
                    train_val["train"].shuffle().map(
                        partial(generate_and_tokenize_prompt, tune_config.tune_dataset.name,
                                tune_config.tune_dataset.train_on_inputs, tokenizer, tune_config.cutoff_len,
                                model_config.name))
                )
                val_data = (
                    train_val["test"].shuffle().map(
                        partial(generate_and_tokenize_prompt, tune_config.tune_dataset.name,
                                tune_config.tune_dataset.train_on_inputs, tokenizer, tune_config.cutoff_len,
                                model_config.name))
                )
            else:
                train_data = data["train"].shuffle().map(
                    partial(generate_and_tokenize_prompt, tune_config.tune_dataset.name,
                            tune_config.tune_dataset.train_on_inputs, tokenizer, tune_config.cutoff_len,
                            model_config.name))
                val_data = None
            with _cache_file(f'{tune_config.tune_dataset.name}_train_data.pkl').open('wb') as f:
                pickle.dump(train_data, f)
            with _cache_file(f'{tune_config.tune_dataset.name}_val_data.pkl').open('wb') as f:
                pickle.dump(val_data, f)
        else:
            if tune_config.tune_dataset.split:
                train_data = (
                    get_dataset[tune_config.tune_dataset.name](tokenizer=tokenizer,
                                                               file_path=tune_config.tune_dataset.train_path,
                                                               block_size=tune_config.tune_dataset.block_size)
                )
                val_data = (
                    get_dataset[tune_config.tune_dataset.name](tokenizer=tokenizer,
                                                               file_path=tune_config.tune_dataset.eval_path,
                                                               block_size=tune_config.tune_dataset.block_size)
                )
            else:
                raise NotImplementedError
            with _cache_file(f'{tune_config.tune_dataset.name}_train_data.pkl').open('wb') as f:
                pickle.dump(train_data, f)
            with _cache_file(f'{tune_config.tune_dataset.name}_val_data.pkl').open('wb') as f:
                pickle.dump(val_data, f)
    return train_data, val_data


def make_prune_data(prune_config, model_config, seed, tokenizer, model=None):
    print("loading calibdation data")
    if prune_config.prune_dataset.pickle_dump:
        if os.path.exists(prune_config.prune_dataset.pickle.prune_path):
            with open(prune_config.prune_dataset.pickle.prune_path, 'rb') as f:
                dataloader = pickle.load(f)
        else:
            raise FileNotFoundError
    else:
        if prune_config.prune_dataset.type in ['downstream']:
            dataloader, _ = data_prune_ds.get_loaders(prune_config.prune_dataset.name,
                                                      seed=seed, tokenizer=tokenizer,
                                                      total_budget=prune_config.prune_dataset.total_budget)
        else:
            dataloader, _ = data_prune.get_loaders(prune_config.prune_dataset.name,
                                                   nsamples=prune_config.prune_dataset.n_samples,
                                                   seed=seed, seqlen=prune_config.prune_dataset.seq_len,
                                                   tokenizer=tokenizer,
                                                   data_path=prune_config.prune_dataset.path,
                                                   base_model=model_config.name)
        with _cache_file(f'{prune_config.prune_dataset.name}_prune_data.pkl').open('wb') as f:
            pickle.dump(dataloader, f)

    print("dataset loading complete")
    return dataloader
