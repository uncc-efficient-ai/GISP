"""Restore and evaluate selected checkpoints from a GISP pruning trajectory.

This post-processing experiment produces the calibration-budget trajectory in
Table 15. It does not compute importance or produce a new pruning trajectory.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import logging
from pathlib import Path
import re

import torch

from modules.eval.setup_eval import eval_lm_eval
from pruners.non_uniform_pruner import non_uniform_pruner
from pruners.utils import find_layers


logger = logging.getLogger(__name__)
_SPARSITY_NAME = re.compile(
    r"^sp_([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\.pth$"
)


class CheckpointTrajectoryEvaluator(non_uniform_pruner):
    """Physically restore and evaluate every selected trajectory mask."""

    def __init__(self, model, config, data):
        super().__init__(model, config, data)
        self._dense_model = self._back_up_dense_model()

    def prune(self):
        func_name = self.config.task.prune.func_name
        if func_name != "evaluate_checkpoint_trajectory":
            raise ValueError(
                "CheckpointTrajectoryEvaluator requires "
                "task.prune.func_name=evaluate_checkpoint_trajectory; "
                f"got {func_name!r}."
            )

        self.before_pruning()
        checkpoint_dir = self.config.task.prune.restore_config.checkpoint_path
        checkpoint_paths = self._discover_checkpoints(checkpoint_dir)
        for checkpoint_path in checkpoint_paths:
            self._restore_checkpoint(checkpoint_path)
            self.get_model_config().use_cache = self.use_cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            sparsity = self.check_sparsity(real_pruning=True)
            eval_lm_eval(
                self.get_wrapped_model(), self.tokenizer, self.config,
                f"sp_{sparsity}_lm_eval", quick=False,
            )

        # prune_task otherwise evaluates the arbitrary last checkpoint again.
        self.config.evaluation.lm_eval = False
        self.config.evaluation.ppl = False
        self.config.evaluation.commonsense_eval = False

    @staticmethod
    def _checkpoint_sort_key(path: Path):
        match = _SPARSITY_NAME.match(path.name)
        return (0, float(match.group(1)), path.name) if match else (1, path.name)

    @classmethod
    def _discover_checkpoints(cls, checkpoint_dir):
        directory = Path(checkpoint_dir).expanduser()
        if not directory.exists():
            raise FileNotFoundError(
                f"Trajectory checkpoint directory does not exist: {directory}"
            )
        if not directory.is_dir():
            raise NotADirectoryError(
                f"Trajectory checkpoint path is not a directory: {directory}"
            )
        checkpoints = sorted(directory.glob("*.pth"), key=cls._checkpoint_sort_key)
        if not checkpoints:
            raise FileNotFoundError(
                f"No .pth trajectory checkpoints found in: {directory}"
            )
        return checkpoints

    def _back_up_dense_model(self):
        original_device = self.get_model().device
        self.get_model().to("cpu")
        dense_model = copy.deepcopy(self.model)
        self.get_model().to(original_device)
        return dense_model

    def _restore_dense_model(self):
        if self._dense_model is None:
            raise RuntimeError("Dense-model backup is unavailable.")
        original_device = self.get_model().device
        self.get_model().to("cpu")
        self.model = copy.deepcopy(self._dense_model)
        self.get_model().to(original_device)

    @staticmethod
    def _validate_checkpoint(checkpoint, checkpoint_path):
        if not isinstance(checkpoint, Mapping):
            raise TypeError(
                f"Trajectory checkpoint must contain a mapping: {checkpoint_path}"
            )
        if "actual_mask" not in checkpoint:
            raise KeyError(
                f"Trajectory checkpoint is missing required key 'actual_mask': "
                f"{checkpoint_path}"
            )
        mask = checkpoint["actual_mask"]
        if not isinstance(mask, Mapping) or not mask:
            raise ValueError(
                f"Checkpoint 'actual_mask' must be a non-empty mapping: "
                f"{checkpoint_path}"
            )
        return mask

    @staticmethod
    def _keep_mask(pruned, size, device, label):
        if not isinstance(pruned, torch.Tensor):
            raise TypeError(f"Mask for {label} must be a torch.Tensor.")
        pruned = pruned.detach().flatten().to(device)
        if pruned.dtype == torch.bool:
            if pruned.numel() != size:
                raise ValueError(
                    f"Boolean mask for {label} has {pruned.numel()} entries; "
                    f"expected {size}."
                )
            return ~pruned
        if pruned.dtype not in {
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
        }:
            raise TypeError(
                f"Index mask for {label} must use an integer dtype; got {pruned.dtype}."
            )
        pruned = pruned.to(torch.long)
        if pruned.numel() and (pruned.min().item() < 0 or pruned.max().item() >= size):
            raise IndexError(f"Mask for {label} contains an out-of-range index.")
        keep = torch.ones(size, dtype=torch.bool, device=device)
        keep[pruned] = False
        return keep

    @staticmethod
    def _slice_linear_output(linear, keep):
        keep = keep.to(linear.weight.device)
        linear.weight = torch.nn.Parameter(linear.weight.data[keep].clone())
        linear.out_features = int(keep.sum().item())
        if linear.bias is not None:
            linear.bias = torch.nn.Parameter(linear.bias.data[keep].clone())

    @staticmethod
    def _slice_linear_input(linear, keep):
        keep = keep.to(linear.weight.device)
        linear.weight = torch.nn.Parameter(linear.weight.data[:, keep].clone())
        linear.in_features = int(keep.sum().item())

    def _restore_checkpoint(self, checkpoint_path):
        self.save_helper.clean()
        logger.info("Checkpoint data loaded from %s", checkpoint_path)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        masks = self._validate_checkpoint(checkpoint, checkpoint_path)
        self._restore_dense_model()

        model_config = self.get_model().config
        num_heads = model_config.num_attention_heads
        if model_config.num_key_value_heads < num_heads:
            raise NotImplementedError(
                "CheckpointTrajectoryEvaluator is validated for the Llama2 "
                "non-GQA Table-15 experiment only; GQA restoration is unsupported."
            )
        head_dim = model_config.hidden_size // num_heads
        for layer_index, layer in enumerate(self.get_layers()):
            self._prune_attention(layer, layer_index, masks, num_heads, head_dim)
            self._prune_mlp(layer, layer_index, masks)
        return checkpoint

    def _prune_attention(self, layer, layer_index, masks, num_heads, head_dim):
        mapping = self.layer_mapping["attn"]
        mask_key = f"{layer_index}.{mapping['block']}"
        if mask_key not in masks:
            return
        attention = getattr(layer, mapping["block"])
        projections = find_layers(layer)
        q_projection = projections[mapping["q"]]
        head_keep = self._keep_mask(
            masks[mask_key], num_heads, q_projection.weight.device, mask_key
        )
        channel_keep = head_keep.repeat_interleave(head_dim)
        new_num_heads = int(head_keep.sum().item())
        if new_num_heads == 0:
            raise ValueError(f"Checkpoint prunes every attention head in {mask_key}.")
        for name in (mapping["q"], mapping["k"], mapping["v"]):
            self._slice_linear_output(projections[name], channel_keep)
        self._slice_linear_input(projections[mapping["o"]], channel_keep)
        for attribute in ("num_heads", "num_attention_heads", "num_key_value_heads"):
            if hasattr(attention, attribute):
                setattr(attention, attribute, new_num_heads)
        if hasattr(attention, "num_key_value_groups"):
            attention.num_key_value_groups = 1
        if hasattr(attention, "hidden_size"):
            attention.hidden_size = new_num_heads * head_dim

    def _prune_mlp(self, layer, layer_index, masks):
        mapping = self.layer_mapping["mlp"]
        mask_key = f"{layer_index}.{mapping['block']}"
        if mask_key not in masks:
            return
        projections = find_layers(layer)
        up = projections[mapping["u"]]
        channel_keep = self._keep_mask(
            masks[mask_key], up.out_features, up.weight.device, mask_key
        )
        if not channel_keep.any():
            raise ValueError(f"Checkpoint prunes every MLP channel in {mask_key}.")
        self._slice_linear_output(up, channel_keep)
        if "g" in mapping:
            self._slice_linear_output(projections[mapping["g"]], channel_keep)
        self._slice_linear_input(projections[mapping["d"]], channel_keep)
        mlp = getattr(layer, mapping["block"])
        if hasattr(mlp, "intermediate_size"):
            mlp.intermediate_size = int(channel_keep.sum().item())

    def step(self):
        return None

    def get_imps(self):
        return {}

    def finishing_pruning(self, real_pruning=True):
        return None
