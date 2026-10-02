# Third-party notices

GISP's original contributions are licensed under Apache-2.0. The components
listed below retain their upstream copyright notices and license terms.
Source locations are relative to the directory containing `main.py`.

## Hugging Face Transformers

- Upstream: [huggingface/transformers](https://github.com/huggingface/transformers).
- License: [Apache-2.0](licenses/transformers-Apache-2.0.txt).
- Locations: `external_code/GISP/modeling/modeling_llama.py`,
  `modeling_mistral.py`, and `modeling_gpt2.py`, and adapted attention routines
  in `external_code/GISP/pruners/`.
- Copyright 2018- The Hugging Face team. All rights reserved.
- Llama implementation: Copyright 2022 EleutherAI and the HuggingFace Inc.
  team. All rights reserved.
- Mistral implementation: Copyright 2023 Mistral AI and the HuggingFace Inc.
  team. All rights reserved.
- GPT-2 implementation: Copyright 2018 The OpenAI Team Authors and
  HuggingFace Inc. team; Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

The model implementations and attention routines have been modified for
GISP's pruning and mask-handling workflows. Upstream model-file headers are
preserved in the model implementations. The bundled Transformers license
text is from release v4.40.2.

## Wanda

- Upstream: [locuslab/wanda](https://github.com/locuslab/wanda).
- License: [MIT](licenses/wanda-MIT.txt).
- Copyright (c) 2023 CMU Locus Lab.
- Locations: activation-statistics wrappers in
  `external_code/GISP/pruners/layerwrapper.py` and related pruning helpers.

The activation-statistics code has been adapted for GISP's experiment pipeline.
The structured Wanda baseline also uses FLAP adaptations described below.

## FLAP

- Upstream: [CASIA-LMC-Lab/FLAP](https://github.com/CASIA-LMC-Lab/FLAP).
- License: [Apache-2.0](licenses/flap-Apache-2.0.txt).
- Locations: `external_code/GISP/pruners/flap_sp.py`, the structured Wanda
  baseline in `wanda_sp.py`, and helpers in `layerwrapper.py` and `utils.py`.

These portions adapt FLAP's pruning, activation-statistics, and structural
compression code to GISP's configuration, model handling, and pruning records.

## OWL

- Upstream: [luuyin/OWL](https://github.com/luuyin/OWL).
- License: [MIT](licenses/owl-MIT.txt).
- Copyright (c) 2024 Lu Yin.
- Location: outlier-statistics code in `external_code/GISP/pruners/owl_sp.py`.

The outlier-statistics code has been incorporated into the structured OWL baseline.

## EleutherAI Language Model Evaluation Harness

- Upstream: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- License: MIT; the upstream license is included as `LICENSE.md` in the
  vendored `modules/eval/lm-evaluation-harness/` directory.
- Copyright (c) 2020 EleutherAI.

The vendored package declares version 0.4.2. Its license text is preserved
from the upstream v0.4.2 release. Existing notices within individual files
continue to apply.

## Paper figures

Figures 1 and 2 in the README are provided under CC BY 4.0, as in the arXiv
paper. See [assets/NOTICE](assets/NOTICE) for the authors, source, license,
and file mapping.

## Models, datasets, and installed dependencies

Pretrained model weights, datasets, and separately installed dependencies
retain their respective licenses. The GISP code license does not replace
those terms.
