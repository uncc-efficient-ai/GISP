import logging

logger = logging.getLogger(__name__)



def tokenize(prompt, tokenizer, cutoff_len, base_model, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
            result["input_ids"][-1] != tokenizer.eos_token_id
            and len(result["input_ids"]) < cutoff_len
            and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        if "chatglm" not in base_model:
            result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    if "chatglm" in base_model:
        return {"input_ids": result["input_ids"], "labels": result["labels"]}
    else:
        return result


def generate_and_tokenize_prompt(dataset, train_on_inputs, tokenizer, cutoff_len, base_model, data_point):
    if 'c4' in dataset:
        full_prompt = data_point["inputs"]
    else:
        full_prompt = generate_prompt(data_point)
    # if dataset != 'alpaca':
    #     full_prompt = data_point["inputs"]
    # else:
    #     full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len, base_model)
    if not train_on_inputs:
        user_prompt = generate_prompt({**data_point, "output": ""})
        tokenized_user_prompt = tokenize(user_prompt, tokenizer, cutoff_len, base_model, add_eos_token=False)
        user_prompt_len = len(tokenized_user_prompt["input_ids"])

        tokenized_full_prompt["labels"] = [
                                              -100
                                          ] * user_prompt_len + tokenized_full_prompt["labels"][
                                                                user_prompt_len:
                                                                ]  # could be sped up, probably
    return tokenized_full_prompt


def generate_prompt(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. 

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""  # noqa: E501


# !!!! there shouldn't have /n here
def generate_prompt_eval(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response:
                """  # noqa: E501
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request. 

                ### Instruction:
                {instruction}

                ### Response:"""  # noqa: E501
