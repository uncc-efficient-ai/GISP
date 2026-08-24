import logging
import math
import os
import re

import numpy as np
import random
import torch
import tqdm
from datasets import load_dataset, DownloadMode, concatenate_datasets, load_from_disk, Dataset, DatasetDict

from modules.data.utils import generate_prompt, tokenize

logger = logging.getLogger(__name__)


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


def estimate_seqlen_boolq(ds, tokenizer, percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    lengths = []

    for i in idxs[:sample_size]:
        item = ds[i]
        passage = item["passage"]
        question = item["question"].strip().rstrip("?") + "?"
        marker = f"\nQuestion: {question}\nAnswer:"

        pass_ids = tokenizer(passage, add_special_tokens=False).input_ids
        mark_ids = tokenizer(marker, add_special_tokens=False).input_ids
        base_len = len(pass_ids) + len(mark_ids)

        for opt in ("Yes", "No"):
            ans_ids = tokenizer(opt, add_special_tokens=False).input_ids
            lengths.append(base_len + len(ans_ids))

    return int(np.percentile(lengths, percentile))


def estimate_seqlen_arc_easy(ds, tokenizer, percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    lengths = []
    if sample_size == 0:
        sample_size = len(idxs)
    for i in idxs[:sample_size]:
        ex = ds[i]
        q = ex["question"].strip()
        prompt_base = f"Question: {q}\nAnswer:"
        base_ids = tokenizer(prompt_base, add_special_tokens=False).input_ids
        Lb = len(base_ids)
        options = [t.strip() for t in ex["choices"]["text"]]
        for opt in options:
            opt_ids = tokenizer(opt, add_special_tokens=False).input_ids
            lengths.append(Lb + len(opt_ids))

    return int(np.percentile(lengths, percentile))


def estimate_seqlen_arc_challenge(ds, tokenizer, percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)

    lengths = []
    if sample_size == 0:
        sample_size = len(idxs)
    for i in idxs[:sample_size]:
        ex = ds[i]
        q = ex["question"].strip()
        prompt_base = f"Question: {q}\nAnswer:"
        base_ids = tokenizer(prompt_base, add_special_tokens=False).input_ids
        Lb = len(base_ids)
        options = [t.strip() for t in ex["choices"]["text"]]
        for opt in options:
            opt_ids = tokenizer(opt, add_special_tokens=False).input_ids
            lengths.append(Lb + len(opt_ids))

    return int(np.percentile(lengths, percentile))


def preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def estimate_seqlen_hellaswag(ds, tokenizer, percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if sample_size == 0:
        sample_size = len(idxs)
    lengths = []

    for i in idxs[:sample_size]:
        item = ds[i]
        ctx_a = item["ctx_a"]
        ctx_b = item["ctx_b"]
        full_ctx = ctx_a + (" " + ctx_b.capitalize() if ctx_b else "")
        full_ctx = preprocess(full_ctx)
        label_text = preprocess(item["activity_label"])
        prompt = f"{label_text}: {full_ctx}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        Lq_orig = len(prompt_ids)
        endings = [preprocess(e) for e in item["endings"]]
        for ans in endings:
            ans_ids = tokenizer(ans, add_special_tokens=False).input_ids
            lengths.append(Lq_orig + len(ans_ids))

    return int(np.percentile(lengths, percentile))


def estimate_seqlen_openbookqa(ds, tokenizer,
                               percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if sample_size == 0:
        sample_size = len(idxs)

    lengths = []
    for i in idxs[:sample_size]:
        ex = ds[i]
        q = ex["question_stem"].strip()
        base_ids = tokenizer(q, add_special_tokens=False).input_ids
        Lb = len(base_ids)
        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        for txt in texts:
            ans = txt.strip()
            aid = tokenizer(ans, add_special_tokens=False).input_ids
            lengths.append(Lb + len(aid))

    return int(np.percentile(lengths, percentile))


def estimate_seqlen_winogrande(ds, tokenizer,
                               percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if sample_size == 0:
        sample_size = len(idxs)

    lengths = []
    for i in idxs[:sample_size]:
        ex = ds[i]
        sent = ex["sentence"].strip()
        cut = sent.index("_")
        prefix = sent[:cut]
        suffix = sent[cut + 1:].strip()
        options = [ex["option1"].strip(), ex["option2"].strip()]
        suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
        La = len(suffix_ids)

        for choice in options:
            filled = prefix + choice
            pref_ids = tokenizer(filled, add_special_tokens=False).input_ids
            lengths.append(len(pref_ids) + La)

    return int(np.percentile(lengths, percentile))


def estimate_seqlen_piqa(ds, tokenizer,
                         percentile=99, sample_size=1000, seed=0):
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    if sample_size == 0:
        sample_size = len(idxs)

    lengths = []
    for i in idxs[:sample_size]:
        ex = ds[i]
        goal = ex["goal"].strip()
        prompt_base_ids = tokenizer(f"Question: {goal}\nAnswer:",
                                    add_special_tokens=False).input_ids
        Lb = len(prompt_base_ids)
        sol1 = ex["sol1"].strip()
        sol2 = ex["sol2"].strip()
        for ans in (sol1, sol2):
            aid = tokenizer(ans, add_special_tokens=False).input_ids
            lengths.append(Lb + len(aid))

    return int(np.percentile(lengths, percentile))


def estimate_seqlen_medqa(ds, tokenizer,
                          percentile=99, sample_size=1000, seed=0):
    """
    Estimate sequence length for MedQA dataset. The prompt format follows the
    LM evaluation harness: "Question: {sent1}\nA. {ending0}\nB. {ending1}\nC. {ending2}\nD. {ending3}\nAnswer:"
    where the model then predicts a single option letter (A, B, C, or D).
    """
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    lengths = []
    if sample_size == 0:
        sample_size = len(idxs)
    option_letters = ["A", "B", "C", "D"]
    for i in idxs[:sample_size]:
        ex = ds[i]
        question = ex["sent1"].strip()
        endings = [ex["ending0"].strip(), ex["ending1"].strip(), ex["ending2"].strip(), ex["ending3"].strip()]
        # Build the base prompt
        answers = "".join([f"{letter}. {opt}\n" for letter, opt in zip(option_letters, endings)])
        prompt = f"Question: {question}\n{answers}Answer:"
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        Lp_orig = len(p_ids)
        for letter in option_letters:
            a_ids = tokenizer(letter, add_special_tokens=False).input_ids
            lengths.append(Lp_orig + len(a_ids))
    return int(np.percentile(lengths, percentile))


def get_boolq(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("boolq", split="train", trust_remote_code=True).shuffle(seed)
    seqlen = estimate_seqlen_boolq(ds, tokenizer, percentile=99, sample_size=2000, seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (boolq 99th pct)")

    trainloader = []
    total_tokens = 0

    for item in ds:
        passage = item["passage"]
        question = item["question"].strip().rstrip("?") + "?"
        marker = f"\nQuestion: {question}\nAnswer:"
        pass_enc = tokenizer(passage, return_tensors="pt", add_special_tokens=False)
        pass_ids = pass_enc.input_ids  # [1, Lp]
        Lp = pass_ids.size(1)

        mark_ids = tokenizer(marker, add_special_tokens=False).input_ids
        Lm = len(mark_ids)

        options = ["Yes", "No"]
        correct_idx = 0 if item["answer"] else 1

        per_opt_lengths = []
        trimmed_pass_ids = None
        for opt in options:
            ans_ids = tokenizer(opt, add_special_tokens=False).input_ids
            La = len(ans_ids)
            if La >= seqlen:
                per_opt_lengths = None
                break
            max_pass = seqlen - (Lm + La)
            if Lp > max_pass:
                this_pass_ids = pass_ids[:, -max_pass:]
                this_Lp = max_pass
            else:
                this_pass_ids = pass_ids
                this_Lp = Lp
            per_opt_lengths.append(this_Lp + Lm + La)
            trimmed_pass_ids = this_pass_ids

        if per_opt_lengths is None:
            continue

        needed = sum(per_opt_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, opt in enumerate(options):
            ans_ids = tokenizer(opt, add_special_tokens=False).input_ids
            La = len(ans_ids)

            inp = torch.cat([
                trimmed_pass_ids,
                torch.tensor([mark_ids], dtype=torch.long),
                torch.tensor([ans_ids], dtype=torch.long)
            ], dim=1)  # [1, L]

            labels = inp.clone()
            labels[0, : (trimmed_pass_ids.size(1) + Lm)] = -100

            pad_len = seqlen - (trimmed_pass_ids.size(1) + Lm + La)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed
    if total_tokens < max_tokens:
        logger.info(f"Warning: task boolq only generated {total_tokens} tokens, budget was {max_tokens}")
    valdata = load_dataset("boolq", split="validation", trust_remote_code=True)
    val_prompts = [
        f"{item['passage']}\nQuestion: {item['question'].strip().rstrip('?')}?\nAnswer:"
        for item in valdata
    ]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(
        val_text,
        return_tensors="pt",
        truncation=True,
        max_length=256 * seqlen
    ).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_arc_easy(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy",
                      split="train", trust_remote_code=True).shuffle(seed)

    seqlen = estimate_seqlen_arc_easy(ds, tokenizer,
                                      percentile=99,
                                      sample_size=2000,
                                      seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (arc_easy 99th pct)")

    trainloader = []
    total_tokens = 0

    for ex in ds:
        q_text = ex["question"].strip()
        options = [t.strip() for t in ex["choices"]["text"]]
        correct_idx = ex["choices"]["label"].index(ex["answerKey"])

        prompt_base = f"Question: {q_text}\nAnswer:"
        q_enc = tokenizer(prompt_base, return_tensors="pt", add_special_tokens=False)
        q_ids = q_enc.input_ids  # [1, Lq_orig]
        Lq_orig = q_ids.size(1)

        per_opt_lengths = []
        trimmed_qs = []
        for opt in options:
            a_ids = tokenizer(opt, add_special_tokens=False).input_ids
            La = len(a_ids)
            if La >= seqlen:
                per_opt_lengths = None
                break
            max_q = seqlen - La
            Lq_eff = min(Lq_orig, max_q)
            per_opt_lengths.append(Lq_eff + La)
            trimmed_qs.append(q_ids[:, -Lq_eff:] if Lq_orig > max_q else q_ids)

        if per_opt_lengths is None:
            continue

        needed = sum(per_opt_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, (opt, length_i, q_trim) in enumerate(zip(options, per_opt_lengths, trimmed_qs)):
            a_ids = tokenizer(opt, add_special_tokens=False).input_ids
            inp = torch.cat([q_trim, torch.tensor([a_ids], dtype=torch.long)], dim=1)
            labels = inp.clone()
            labels[0, :q_trim.size(1)] = -100

            pad_len = seqlen - length_i
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed
    if total_tokens < max_tokens:
        logger.info(f"Warning: task arc_easy only generated {total_tokens} tokens, budget was {max_tokens}")
    val_prompts = [
        f"Question: {ex['question'].strip()}\nAnswer:"
        for ex in load_dataset("allenai/ai2_arc", "ARC-Easy",
                               split="validation", trust_remote_code=True)
    ]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_arc_challenge(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train", trust_remote_code=True).shuffle(seed)

    seqlen = estimate_seqlen_arc_challenge(ds, tokenizer, 99, 2000, seed)
    logger.info(f"[Auto] set seqlen={seqlen} (arc_challenge 99th pct)")
    trainloader = []
    total_tokens = 0

    for ex in ds:
        q_text = ex["question"].strip()
        options = [t.strip() for t in ex["choices"]["text"]]
        correct_idx = ex["choices"]["label"].index(ex["answerKey"])
        prompt_base = f"Question: {q_text}\nAnswer:"
        q_enc = tokenizer(prompt_base, return_tensors="pt", add_special_tokens=False)
        q_ids = q_enc.input_ids  # [1, Lq_orig]
        Lq_orig = q_ids.size(1)

        per_opt_lengths = []
        trimmed_qs = []
        for opt in options:
            a_ids = tokenizer(opt, add_special_tokens=False).input_ids
            La = len(a_ids)
            if La >= seqlen:
                per_opt_lengths = None
                break
            max_q = seqlen - La
            Lq_eff = min(Lq_orig, max_q)
            per_opt_lengths.append(Lq_eff + La)
            trimmed_qs.append(q_ids[:, -Lq_eff:] if Lq_orig > max_q else q_ids)

        if per_opt_lengths is None:
            continue

        needed = sum(per_opt_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, (opt, length_i, q_trim) in enumerate(zip(options, per_opt_lengths, trimmed_qs)):
            a_ids = tokenizer(opt, add_special_tokens=False).input_ids
            inp = torch.cat([q_trim, torch.tensor([a_ids], dtype=torch.long)], dim=1)
            labels = inp.clone()
            labels[0, :q_trim.size(1)] = -100

            pad_len = seqlen - length_i
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed

    if total_tokens < max_tokens:
        logger.info(f"Warning: task arc_challenge only generated {total_tokens} tokens, budget was {max_tokens}")

    val_prompts = [f"Question: {ex['question'].strip()}\nAnswer:" for ex in
                   load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation", trust_remote_code=True)]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text, return_tensors="pt", truncation=True, max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_hellaswag(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("hellaswag", split="train", trust_remote_code=True).shuffle(seed)

    seqlen = estimate_seqlen_hellaswag(ds, tokenizer,
                                       percentile=99,
                                       sample_size=2000,
                                       seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (HellaSwag 99th pct)")

    trainloader = []
    total_tokens = 0

    for item in ds:
        ctx_a = item["ctx_a"]
        ctx_b = item["ctx_b"]
        full_ctx = ctx_a + (" " + ctx_b.capitalize() if ctx_b else "")
        full_ctx = preprocess(full_ctx)
        label_text = preprocess(item["activity_label"])
        prompt = f"{label_text}: {full_ctx}"

        # tokenize prompt
        q_ids = tokenizer(prompt, return_tensors="pt",
                          add_special_tokens=False).input_ids  # [1, Lq_orig]
        Lq_orig = q_ids.size(1)

        endings = [preprocess(e) for e in item["endings"]]
        correct_idx = int(item["label"])
        per_lengths = []
        trimmed_q_ids = None
        for ending in endings:
            ans_ids = tokenizer(ending, add_special_tokens=False).input_ids
            La = len(ans_ids)
            if La >= seqlen:
                per_lengths = None
                break
            max_q = seqlen - La
            if Lq_orig > max_q:
                this_q = q_ids[:, -max_q:]
                this_Lq = max_q
            else:
                this_q = q_ids
                this_Lq = Lq_orig
            per_lengths.append(this_Lq + La)
            trimmed_q_ids = this_q

        if per_lengths is None:
            continue

        needed = sum(per_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, ending in enumerate(endings):
            ans_ids = tokenizer(ending, add_special_tokens=False).input_ids
            La = len(ans_ids)

            inp = torch.cat([
                trimmed_q_ids,
                torch.tensor([ans_ids], dtype=torch.long)
            ], dim=1)  # [1, Lq_eff + La]

            labels = inp.clone()
            labels[0, :trimmed_q_ids.size(1)] = -100

            pad_len = seqlen - (trimmed_q_ids.size(1) + La)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed

    if total_tokens < max_tokens:
        logger.info(f"Warning: task hellaswag only generated {total_tokens} tokens, budget was {max_tokens}")

    valdata = load_dataset("hellaswag", split="validation", trust_remote_code=True)
    val_prompts = []
    for itm in valdata:
        ctx_a = itm["ctx_a"]
        ctx_b = itm["ctx_b"]
        full_ctx = ctx_a + (" " + ctx_b.capitalize() if ctx_b else "")
        full_ctx = preprocess(full_ctx)
        label_text = preprocess(itm["activity_label"])
        val_prompts.append(f"{label_text}: {full_ctx}")
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(
        val_text,
        return_tensors="pt",
        truncation=True,
        max_length=256 * seqlen
    ).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_openbookqa(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("allenai/openbookqa", "main", split="train", trust_remote_code=True).shuffle(seed)

    seqlen = estimate_seqlen_openbookqa(ds, tokenizer,
                                        percentile=99,
                                        sample_size=2000,
                                        seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (OpenBookQA 99th pct)")

    trainloader = []
    total_tokens = 0

    for ex in ds:
        # prompt_base = question_stem
        q_text = ex["question_stem"].strip()
        q_enc = tokenizer(q_text, return_tensors="pt", add_special_tokens=False)
        q_ids = q_enc.input_ids  # [1, Lq_orig]
        Lq_orig = q_ids.size(1)

        labels = ex["choices"]["label"]
        texts = ex["choices"]["text"]
        correct_idx = labels.index(ex["answerKey"].lstrip())
        per_lengths = []
        trimmed_q_ids = None
        for txt in texts:
            ans = txt.strip()
            aid = tokenizer(ans, add_special_tokens=False).input_ids
            La = len(aid)
            if La >= seqlen:
                per_lengths = None
                break
            max_q = seqlen - La
            if Lq_orig > max_q:
                this_q = q_ids[:, -max_q:]
                this_Lq = max_q
            else:
                this_q = q_ids
                this_Lq = Lq_orig
            per_lengths.append(this_Lq + La)
            trimmed_q_ids = this_q

        if per_lengths is None:
            continue

        needed = sum(per_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, txt in enumerate(texts):
            ans = txt.strip()
            aid = tokenizer(ans, add_special_tokens=False).input_ids
            La = len(aid)

            inp = torch.cat([
                trimmed_q_ids,
                torch.tensor([aid], dtype=torch.long)
            ], dim=1)  # [1, Lq_eff + La]

            labels_tensor = inp.clone()
            labels_tensor[0, : trimmed_q_ids.size(1)] = -100

            pad_len = seqlen - (trimmed_q_ids.size(1) + La)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels_tensor = torch.cat([labels_tensor, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels_tensor, is_positive))

        total_tokens += needed

    if total_tokens < max_tokens:
        logger.info(f"Warning: task openbookqa only generated {total_tokens} tokens, budget was {max_tokens}")

    val_ds = load_dataset("allenai/openbookqa", "main", split="validation", trust_remote_code=True)
    val_prompts = [ex["question_stem"].strip() for ex in val_ds]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_winogrande(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("winogrande", "winogrande_xl", split="train",
                      trust_remote_code=True).shuffle(seed)

    seqlen = estimate_seqlen_winogrande(ds, tokenizer,
                                        percentile=99,
                                        sample_size=2000,
                                        seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (Winogrande XL 99th pct)")

    trainloader = []
    total_tokens = 0

    for ex in ds:
        sent = ex["sentence"].strip()
        cut = sent.index("_")
        prefix = sent[:cut]
        suffix = sent[cut + 1:].strip()
        options = [ex["option1"].strip(), ex["option2"].strip()]
        correct_idx = 0 if ex["answer"] == 1 else 1

        suffix_ids = tokenizer(suffix, add_special_tokens=False).input_ids
        La = len(suffix_ids)
        if La >= seqlen:
            continue

        per_lengths = []
        trimmed_pref = []
        for choice in options:
            filled = prefix + choice
            pref_ids = tokenizer(filled, return_tensors="pt",
                                 add_special_tokens=False).input_ids  # [1, Lp_orig]
            Lp_orig = pref_ids.size(1)

            max_p = seqlen - La
            if Lp_orig > max_p:
                q_trim = pref_ids[:, -max_p:]
                Lp_eff = max_p
            else:
                q_trim = pref_ids
                Lp_eff = Lp_orig

            per_lengths.append(Lp_eff + La)
            trimmed_pref.append(q_trim)

        if any(length > seqlen for length in per_lengths):
            continue

        needed = sum(per_lengths)
        if total_tokens + needed > max_tokens:
            break
        for idx, q_trim in enumerate(trimmed_pref):
            Lp_eff = q_trim.size(1)

            inp = torch.cat([
                q_trim,
                torch.tensor([suffix_ids], dtype=torch.long)
            ], dim=1)  # [1, Lp_eff + La]

            labels = inp.clone()
            labels[0, :Lp_eff] = -100

            pad_len = seqlen - (Lp_eff + La)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed

    if total_tokens < max_tokens:
        logger.info(f"Warning: task winogrande only generated {total_tokens} tokens, budget was {max_tokens}")

    val_ds = load_dataset("winogrande", "winogrande_xl",
                          split="validation", trust_remote_code=True)
    val_texts = [ex["sentence"].strip() for ex in val_ds]
    val_text = " ".join(val_texts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_piqa(max_tokens: int, seed: int, tokenizer):
    random.seed(seed)
    ds = load_dataset("piqa", split="train", trust_remote_code=True).shuffle(seed)
    seqlen = estimate_seqlen_piqa(ds, tokenizer,
                                  percentile=99,
                                  sample_size=2000,
                                  seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (PIQA 99th pct)")

    trainloader = []
    total_tokens = 0

    for ex in ds:
        goal = ex["goal"].strip()
        sol_correct = ex["sol1"].strip() if ex["label"] == 0 else ex["sol2"].strip()
        sol_wrong = ex["sol2"].strip() if ex["label"] == 0 else ex["sol1"].strip()

        prompt = f"Question: {goal}\nAnswer:"
        p_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        p_ids = p_enc.input_ids  # [1, Lb]
        Lb = p_ids.size(1)

        per_lengths = []
        trimmed_prompts = []
        for ans in (sol_correct, sol_wrong):
            aid = tokenizer(ans, add_special_tokens=False).input_ids
            La = len(aid)
            if La >= seqlen:
                per_lengths = None
                break
            max_p = seqlen - La
            if Lb > max_p:
                trim = p_ids[:, -max_p:]
                Lp_eff = max_p
            else:
                trim = p_ids
                Lp_eff = Lb
            per_lengths.append(Lp_eff + La)
            trimmed_prompts.append(trim)

        if per_lengths is None:
            continue

        needed = sum(per_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, trim in enumerate(trimmed_prompts):
            ans = sol_correct if idx == 0 else sol_wrong
            aid = tokenizer(ans, add_special_tokens=False).input_ids
            La = len(aid)

            inp = torch.cat([
                trim,
                torch.tensor([aid], dtype=torch.long)
            ], dim=1)  # [1, Lp_eff+La]

            labels = inp.clone()
            labels[0, :trim.size(1)] = -100

            pad_len = seqlen - (trim.size(1) + La)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)

            is_positive = (idx == 0)
            trainloader.append((inp, labels, is_positive))

        total_tokens += needed

    if total_tokens < max_tokens:
        logger.info(f"Warning: task piqa only generated {total_tokens} tokens, budget was {max_tokens}")

    val_ds = load_dataset("piqa", split="validation", trust_remote_code=True)
    val_prompts = [f"Question: {ex['goal'].strip()}\nAnswer:" for ex in val_ds]
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_medqa(max_tokens: int, seed: int, tokenizer):
    """
    Generate training data and validation encodings for the MedQA task. The
    MedQA task is a multiple-choice question answering dataset where each
    question includes four answer options. Following the LM evaluation harness,
    the prompt lists the question and all options with labels A-D, and the model
    must predict the correct option letter. The training loader pairs the
    truncated prompt with each possible option letter and marks the correct
    letter as positive.
    """
    random.seed(seed)
    ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="train", trust_remote_code=True).shuffle(seed)
    seqlen = estimate_seqlen_medqa(ds, tokenizer, percentile=99, sample_size=2000, seed=seed)
    logger.info(f"[Auto] set seqlen={seqlen} (MedQA 99th pct)")

    trainloader = []
    total_tokens = 0
    option_letters = ["A", "B", "C", "D"]
    for ex in ds:
        # Construct the full prompt with question and all options
        question = ex["sent1"].strip()
        endings = [ex["ending0"].strip(), ex["ending1"].strip(), ex["ending2"].strip(), ex["ending3"].strip()]
        answers_text = "".join([f"{letter}. {opt}\n" for letter, opt in zip(option_letters, endings)])
        prompt = f"Question: {question}\n{answers_text}Answer:"
        p_enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        p_ids = p_enc.input_ids
        Lp_orig = p_ids.size(1)
        correct_idx = int(ex["label"])

        per_lengths = []
        trimmed_p_ids = None
        for letter in option_letters:
            a_ids = tokenizer(letter, add_special_tokens=False).input_ids
            La = len(a_ids)
            if La >= seqlen:
                per_lengths = None
                break
            max_p = seqlen - La
            if Lp_orig > max_p:
                this_p = p_ids[:, -max_p:]
                this_Lp = max_p
            else:
                this_p = p_ids
                this_Lp = Lp_orig
            per_lengths.append(this_Lp + La)
            trimmed_p_ids = this_p

        if per_lengths is None:
            continue

        needed = sum(per_lengths)
        if total_tokens + needed > max_tokens:
            break

        for idx, letter in enumerate(option_letters):
            a_ids = tokenizer(letter, add_special_tokens=False).input_ids
            La = len(a_ids)
            inp = torch.cat([
                trimmed_p_ids,
                torch.tensor([a_ids], dtype=torch.long)
            ], dim=1)
            labels_tensor = inp.clone()
            labels_tensor[0, :trimmed_p_ids.size(1)] = -100

            pad_len = seqlen - (trimmed_p_ids.size(1) + La)
            if pad_len > 0:
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels_tensor = torch.cat([labels_tensor, pad_lbls], dim=1)

            is_positive = (idx == correct_idx)
            trainloader.append((inp, labels_tensor, is_positive))

        total_tokens += needed

    if total_tokens < max_tokens:
        logger.info(f"Warning: task medqa only generated {total_tokens} tokens, budget was {max_tokens}")

    # Prepare validation encodings
    val_ds = load_dataset("GBaker/MedQA-USMLE-4-options-hf", split="validation", trust_remote_code=True)
    val_prompts = []
    for ex in val_ds:
        question = ex["sent1"].strip()
        endings = [ex["ending0"].strip(), ex["ending1"].strip(), ex["ending2"].strip(), ex["ending3"].strip()]
        answers_text = "".join([f"{letter}. {opt}\n" for letter, opt in zip(option_letters, endings)])
        val_prompts.append(f"Question: {question}\n{answers_text}Answer:")
    val_text = " ".join(val_prompts)
    val_enc = tokenizer(val_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256 * seqlen).input_ids
    valenc = TokenizerWrapper(val_enc)

    return trainloader, valenc, total_tokens


def get_cmqa_no_pad(total_budget: int,
                    seed: int,
                    tokenizer):
    tasks = {
        'boolq': get_boolq,
        'arc_easy': get_arc_easy,
        'arc_challenge': get_arc_challenge,
        'hellaswag': get_hellaswag,
        'openbookqa': get_openbookqa,
        'winogrande': get_winogrande,
        'piqa': get_piqa,
    }

    num_tasks = len(tasks)
    base_budget = total_budget // num_tasks
    budgets = {name: base_budget for name in tasks}

    trainloader = []
    valencs = {}
    for name, fn in tasks.items():
        loader, valenc, _ = fn(budgets[name], seed, tokenizer)
        trainloader.extend(loader)
        valencs[name] = valenc

    return trainloader, valencs


def get_cmqa(total_budget: int, seed: int, tokenizer):
    random.seed(seed)
    tasks = {
        'boolq': get_boolq,
        'arc_easy': get_arc_easy,
        'arc_challenge': get_arc_challenge,
        'hellaswag': get_hellaswag,
        'openbookqa': get_openbookqa,
        'winogrande': get_winogrande,
        'piqa': get_piqa,
    }

    per_budget = total_budget // len(tasks)
    loaders = {}
    valencs = {}
    for name, fn in tasks.items():
        loader, valenc = fn(per_budget, seed, tokenizer)
        loaders[name] = loader
        valencs[name] = valenc

    seqlens = [loader[0][0].size(1) for loader in loaders.values() if loader]
    unified_seqlen = max(seqlens)

    def repad(loader):
        out = []
        for inp, labels in loader:
            cur_len = inp.size(1)
            if cur_len < unified_seqlen:
                pad_len = unified_seqlen - cur_len
                pad_ids = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=torch.long)
                pad_lbls = torch.full((1, pad_len), -100, dtype=torch.long)
                inp = torch.cat([inp, pad_ids], dim=1)
                labels = torch.cat([labels, pad_lbls], dim=1)
            out.append((inp, labels))
        return out

    for name in loaders:
        loaders[name] = repad(loaders[name])

    trainloader = []
    for loader in loaders.values():
        trainloader.extend(loader)

    return trainloader, valencs


def process_on_device(buckets, model, device):
    model.to(device)
    losses = []
    for L, inp, lab in tqdm.tqdm(buckets, desc=f"Estimating on {device}"):
        inp = inp.to(device)
        lab = lab.to(device)
        if inp.dim() == 1:
            inp_batch = inp.unsqueeze(0)
            lab_batch = lab.unsqueeze(0)
        else:
            inp_batch = inp
            lab_batch = lab

        outputs = model(inp_batch, labels=lab_batch)
        valid_tokens = (lab_batch != -100).sum().item()
        loss_sum = outputs.loss * valid_tokens
        losses.append(loss_sum.item())
    return losses


def build_input_and_label_for_task(name, ex, tokenizer):
    if name == 'boolq':
        passage = ex['passage']
        question = ex['question'].strip().rstrip('?') + '?'
        answer = 'Yes' if ex['answer'] else 'No'
        marker = f"\nQuestion: {question}\nAnswer:"
        p_ids = tokenizer(passage, add_special_tokens=False).input_ids
        m_ids = tokenizer(marker, add_special_tokens=False).input_ids
        a_ids = tokenizer(answer, add_special_tokens=False).input_ids
        ids = p_ids + m_ids + a_ids
        labels = [-100] * (len(p_ids) + len(m_ids)) + a_ids

    elif name in ('arc_easy', 'arc_challenge'):
        q = ex['question'].strip()
        labels_list = ex['choices']['label']
        texts = ex['choices']['text']
        ans_key = ex['answerKey']
        idx = labels_list.index(ans_key)
        a = texts[idx].strip()
        prompt = f"Question: {q}\nAnswer:"
        q_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(a, add_special_tokens=False).input_ids
        ids = q_ids + a_ids
        labels = [-100] * len(q_ids) + a_ids

    elif name == 'hellaswag':
        ctx_a = ex['ctx_a']
        ctx_b = ex['ctx_b']
        full = ctx_a + ('' if not ctx_b else ' ' + ctx_b.capitalize())
        full = preprocess(full)
        label_txt = preprocess(ex['activity_label'])
        prompt = f"{label_txt}: {full}"
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        ans = preprocess(ex['endings'][int(ex['label'])])
        a_ids = tokenizer(ans, add_special_tokens=False).input_ids
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids

    elif name == 'openbookqa':
        q = ex['question_stem'].strip()
        labs = ex['choices']['label']
        txts = ex['choices']['text']
        idx = labs.index(ex['answerKey'].lstrip())
        a = txts[idx].strip()
        q_ids = tokenizer(q, add_special_tokens=False).input_ids
        a_ids = tokenizer(a, add_special_tokens=False).input_ids
        ids = q_ids + a_ids
        labels = [-100] * len(q_ids) + a_ids

    elif name == 'winogrande':
        sent = ex['sentence'].strip()
        i = sent.index('_')
        choice = ex['option1'].strip() if ex['answer'] == 1 else ex['option2'].strip()
        filled = sent[:i] + choice
        suffix = sent[i + 1:].strip()
        p_ids = tokenizer(filled, add_special_tokens=False).input_ids
        a_ids = tokenizer(suffix, add_special_tokens=False).input_ids
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids

    elif name == 'piqa':
        goal = ex['goal'].strip()
        sol = ex['sol1'].strip() if ex['label'] == 0 else ex['sol2'].strip()
        prompt = f"Question: {goal}\nAnswer:"
        p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        a_ids = tokenizer(sol, add_special_tokens=False).input_ids
        ids = p_ids + a_ids
        labels = [-100] * len(p_ids) + a_ids

    else:
        raise NotImplementedError(f"Unknown task {name}")

    inp = torch.tensor(ids, dtype=torch.long)
    lab = torch.tensor(labels, dtype=torch.long)
    return inp, lab


def get_loaders(name='arc_easy', seed=0, total_budget=512000, tokenizer=None):
    if "arc_easy" in name:
        trainloader, valenc, total_tokens = get_arc_easy(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "arc_challenge" in name:
        trainloader, valenc, total_tokens = get_arc_challenge(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "boolq" in name:
        trainloader, valenc, total_tokens = get_boolq(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "hellaswag" in name:
        trainloader, valenc, total_tokens = get_hellaswag(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "openbookqa" in name:
        trainloader, valenc, total_tokens = get_openbookqa(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "winogrande" in name:
        trainloader, valenc, total_tokens = get_winogrande(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "piqa" in name:
        trainloader, valenc, total_tokens = get_piqa(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "medqa" in name:
        trainloader, valenc, total_tokens = get_medqa(total_budget, seed, tokenizer)
        return trainloader, valenc
    elif "cmqa_no_pad" in name:
        return get_cmqa_no_pad(total_budget, seed, tokenizer)
    elif "cmqa" in name:
        return get_cmqa(total_budget, seed, tokenizer)
    else:
        raise ValueError
