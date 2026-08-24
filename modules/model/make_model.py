from .hf import make_hf_model,make_bin_model


def make_model(model_config):
    if model_config.struct == 'hf':
        model, tokenizer, config = make_hf_model(model_config)
    elif model_config.struct == 'bin':
        model, tokenizer, config = make_bin_model(model_config)
    return model, tokenizer, config
