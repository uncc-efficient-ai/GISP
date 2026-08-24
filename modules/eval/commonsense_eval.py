import copy
import json
import os
import random
import re

import fire
import numpy as np
import torch
from tqdm import tqdm
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from ..data.utils import generate_prompt_eval


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def recovery(base_model: str = "", resume_from_checkpoint: str = '', seed: int = 0, dataset: str = '',
             batch_size: int = 1,
             output_dir: str = 'outputs'):
    if resume_from_checkpoint != '':
        checkpoint_name = os.path.join(
            resume_from_checkpoint, "model.bin"
        )  # Full checkpoint
        x = torch.load(checkpoint_name)
        tokenizer = x['tokenizer']
        model = x['model'].to('cuda')
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
        )
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = model.to('cuda')
    main(model, tokenizer, seed, dataset, batch_size, output_dir)


def main(model, tokenizer, seed, dataset, batch_size, output_dir):
    set_random_seed(seed)
    model.eval()
    model.half()

    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )
        tokenizer.padding_side = "left"  # Allow batched inference

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    score = {}
    if dataset:
        targets = dataset
    else:
        targets = ['boolq', 'piqa', 'hellaswag', 'winogrande', 'ARC-Easy', 'ARC-Challenge', 'openbookqa', 'social_i_qa']

    for d in targets:
        output_json = f'{d}.json'
        save_file = os.path.join(output_dir, output_json)

        datasets = load_data(d)
        batches = create_batch(datasets, batch_size)

        total = len(batches)
        correct = 0
        current = 0
        output_data = []
        execution_time = AverageMeter()
        pbar = tqdm(total=total)
        for idx, batch in enumerate(batches):
            current += len(batch)
            instructions = [data.get('instruction') for data in batch]

            outputs, latency = run_evaluate(model, tokenizer, instructions, None)

            if idx > 0:
                execution_time.update(latency)

            for data, output in zip(batch, outputs):
                label = data.get('answer')
                flag = False

                predict = extract_answer(d, output)
                predict = predict.lower()

                if label == predict:
                    correct += 1
                    flag = True

                new_data = copy.deepcopy(data)
                new_data['output_pred'] = output
                new_data['pred'] = predict
                new_data['flag'] = flag
                output_data.append(new_data)
                # print(data["instruction"])
                # print(output)
                # print('output:', output)
                # print('prediction:', predict)
                # print('label:', label)

                acc = correct / current

                # if args.visualize:
                #     lat.append(latency)

            print('---------------')
            print(f'\rdataset:{d} | test:{idx + 1}/{total} | accuracy = {acc} | latency = {latency:.3f} (ms)')
            print('---------------')
            with open(save_file, 'w+') as f:
                json.dump(output_data, f, indent=4)
            pbar.update(1)
        pbar.close()
        print('\n')
        print(f'test finished, average latency (ms): {execution_time.avg} | accuracy = {acc} | sd = {False}')
        torch.cuda.empty_cache()
        score[d] = acc

    output_json = f'acc.json'
    with open(os.path.join(output_dir, output_json), 'w+', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=4)
    print(score)


def run_evaluate(
        model,
        tokenizer,
        instructions,
        tiny_model=None,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=32,
        **kwargs
):
    prompts = [generate_prompt_eval(instruction, input) for instruction in instructions]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    # generation_config = GenerationConfig(
    #     temperature=temperature,
    #     top_p=top_p,
    #     top_k=top_k,
    #     num_beams=num_beams,
    #     **kwargs,
    # )

    # generation_config.do_sample = True

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"].to(model.device),
            pad_token_id=tokenizer.pad_token_id,

            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
            # assistant_model=tiny_model,
            # generation_config=generation_config,
        )
        end.record()
        torch.cuda.synchronize()

        s = generation_output.sequences
        outputs = tokenizer.batch_decode(s, skip_special_tokens=True)
        # print(f"outputs: {outputs}")
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]

        latency = start.elapsed_time(end)

    return outputs, latency


def create_batch(dataset, batch_size):
    batches = []
    num_batch = len(dataset) // batch_size if len(dataset) % batch_size == 0 else len(dataset) // batch_size + 1
    for i in range(num_batch):
        batch = dataset[i * batch_size: min((i + 1) * batch_size, len(dataset))]
        batches.append(batch)
    return batches


def extract_answer(dataset, sentence: str) -> float:
    if dataset == 'boolq':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'true|false', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'piqa':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'solution1|solution2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset in ['social_i_qa', 'ARC-Challenge', 'ARC-Easy', 'openbookqa']:
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'answer1|answer2|answer3|answer4|answer5', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'hellaswag':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'ending1|ending2|ending3|ending4', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]
    elif dataset == 'winogrande':
        sentence_ = sentence.strip()
        pred_answers = re.findall(r'option1|option2', sentence_)
        if not pred_answers:
            return ""
        return pred_answers[0]


def load_data(dataset) -> list:
    """
    read data from dataset file
    Args:
        args:

    Returns:

    """
    file_path = f'dataset/{dataset}/test.json'
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can not find dataset file : {file_path}")
    json_data = json.load(open(file_path, 'r'))
    return json_data


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    fire.Fire(recovery)
