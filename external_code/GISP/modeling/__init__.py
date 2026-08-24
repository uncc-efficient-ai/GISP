from .modeling_llama import *
from .modeling_gpt2 import *
from .modeling_mistral import *

def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
    # return {"input_ids": input_ids, 'emb_match': kwargs['emb_match'],
    #         'control_code': kwargs['control_code'], 'past_key_values': kwargs['past_key_values']}

    # only last token for inputs_ids if past is defined in kwargs
    if past:
        input_ids = input_ids[:, -1].unsqueeze(-1)

    # ##########  for batch generation ####################
    # print(kwargs.keys())
    use_prefix_test = kwargs.get("use_prefix_test", False)

    if use_prefix_test:
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create postion_ids on the fly for batch generation
            # print(attention_mask)
            position_ids = attention_mask.long().cumsum(-1) - 1
            # print(position_ids)
            position_ids.masked_fill_(attention_mask == 0, 1)
            # take the equivalent length as the input ids.
            input_len = input_ids.shape[-1]
            position_ids = position_ids[:, -input_len:]
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
    # print(position_ids, attention_mask.shape)
    ##############################
    if past is None:
        # print('only at the beginnning. ')
        if 'past_key_values' in kwargs:
            past = kwargs['past_key_values']
        else:
            past = None

    if use_prefix_test:
        # print('using the batch gen')
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            #############for batch gen########
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            #####################
        }
    else:
        # print('No batch gen')
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
        }


GPT2LMHeadModel.prepare_inputs_for_generation = prepare_inputs_for_generation
