# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""FSDP sharding of Hugging Face models across TPU cores.

Two execution models, matching how TPU hosts are provisioned:

- Single-process hosts (Kaggle, Colab: one process drives all chips): use the
  SPMD variant (SpmdFullyShardedDataParallel) with a mesh over all physical
  cores. Parameters may be bf16. SPMD must already be active (see
  environment.setup_tpu_environment) BEFORE the model is loaded or wrapped.
- Multi-process jobs (one process per core): use classic
  XlaFullyShardedDataParallel, which requires fp32 parameters (compute stays
  bf16 via compute_dtype).

Models must be loaded on CPU before wrapping so the wrapper can shard them,
and LoRA/PEFT adapters should be attached AFTER wrapping (adapter parameters
then stay replicated on every core and can be updated directly).
"""

from functools import partial
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel

from .detection import detect_tpu, is_torch_xla_available
from .device import get_xla_device_count


def fsdp_uses_spmd() -> bool:
    """True when the single-process SPMD FSDP path applies (Kaggle/Colab TPU VMs)."""
    if not (detect_tpu() and is_torch_xla_available()):
        return False
    import torch_xla.runtime as xr

    return xr.process_count() == 1 and get_xla_device_count() > 1


def resolve_fsdp_dtype() -> str:
    """Parameter dtype required by the FSDP variant in use.

    SPMD FSDPv2 accepts bf16 parameters directly; classic FSDP requires fp32
    parameters with bf16 compute.
    """
    return "bfloat16" if fsdp_uses_spmd() else "float32"


def get_fsdp_layer_classes(model: PreTrainedModel) -> set[type[nn.Module]]:
    """Discover transformer layer classes present in the model for FSDP wrapping."""
    return {
        type(module)
        for module in model.modules()
        if "DecoderLayer" in type(module).__name__
        or "TransformerBlock" in type(module).__name__
    }


# Architecture-specific transformer layer class names for FSDP wrapping.
# Used when a config explicitly names layers by class name.
TRANSFORMER_LAYER_CLASSES: dict[str, list[str]] = {
    "LlamaForCausalLM": ["LlamaDecoderLayer"],
    "LlamaForSequenceClassification": ["LlamaDecoderLayer"],
    "CodeLlamaForCausalLM": ["LlamaDecoderLayer"],
    "Qwen2ForCausalLM": ["Qwen2DecoderLayer"],
    "Qwen2MoeForCausalLM": ["Qwen2MoeDecoderLayer"],
    "Qwen3ForCausalLM": ["Qwen3DecoderLayer"],
    "Qwen3MoeForCausalLM": ["Qwen3MoeDecoderLayer"],
    "GemmaForCausalLM": ["GemmaDecoderLayer"],
    "Gemma2ForCausalLM": ["Gemma2DecoderLayer"],
    "Gemma3ForCausalLM": ["Gemma3DecoderLayer"],
    "Gemma3ForConditionalGeneration": ["Gemma3DecoderLayer"],
    "MistralForCausalLM": ["MistralDecoderLayer"],
    "MixtralForCausalLM": ["MixtralDecoderLayer"],
    "Phi3ForCausalLM": ["Phi3DecoderLayer"],
    "Phi4ForCausalLM": ["Phi4DecoderLayer"],
    "YiForCausalLM": ["YiDecoderLayer"],
    "InternLM2ForCausalLM": ["InternLM2DecoderLayer"],
    "InternLM2ForSequenceClassification": ["InternLM2DecoderLayer"],
    "GraniteForCausalLM": ["GraniteDecoderLayer"],
    "GraniteMoeForCausalLM": ["GraniteMoeDecoderLayer"],
    "CohereForCausalLM": ["CohereDecoderLayer"],
    "SmolLM3ForCausalLM": ["SmolLM3DecoderLayer"],
    "DeepseekV2ForCausalLM": ["DeepseekV2DecoderLayer"],
    "DeepseekV3ForCausalLM": ["DeepseekV3DecoderLayer"],
}


def get_fsdp_layer_class_names(model: PreTrainedModel) -> list[str]:
    """Transformer layer class names for FSDP wrapping, by model architecture."""
    model_class = model.__class__.__name__
    return TRANSFORMER_LAYER_CLASSES.get(model_class, [])


def get_default_fsdp_config(model: PreTrainedModel) -> dict[str, Any]:
    """Generate a default FSDP configuration dict for the given model."""
    return {
        "fsdp_transformer_layer_cls_to_wrap": get_fsdp_layer_class_names(model),
    }


def load_fsdp_config(path: str) -> dict[str, Any]:
    """Load FSDP configuration from a JSON file."""
    import json

    with open(path) as f:
        return json.load(f)


def _spmd_shard_output(output: Any, mesh: Any) -> None:
    """Shard the logits output of a wrapped model across the SPMD mesh.

    Required so the XLA compiler implements the FSDP algorithm correctly
    (weights and activations must both be sharded).
    """
    from torch_xla.distributed import spmd

    logits = (
        output
        if isinstance(output, torch.Tensor)
        else getattr(output, "logits", output)
    )
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError(f"Cannot shard output of type {type(output)}")
    spmd.mark_sharding(logits, mesh, (None, None, "fsdp"))


def wrap_model_fsdp(
    model: PreTrainedModel,
    config: dict[str, Any] | str | None = None,
) -> PreTrainedModel:
    """Wrap a (CPU-resident) Hugging Face model with FSDP for TPU parallelism.

    Args:
        model: The model to wrap, loaded on CPU.
        config: FSDP configuration dict, path to a JSON config file, or None
            for auto-configuration.

    Returns:
        FSDP-wrapped model (SPMD variant for single-process hosts, classic
        variant for multi-process jobs).
    """
    if not detect_tpu():
        raise RuntimeError("FSDP wrapping only supported on TPU")
    if not is_torch_xla_available():
        raise RuntimeError("torch_xla not available")

    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy

    if isinstance(config, str):
        config = load_fsdp_config(config)
    elif config is None:
        config = get_default_fsdp_config(model)

    layer_names = config.get("fsdp_transformer_layer_cls_to_wrap", [])
    layer_classes = get_fsdp_layer_classes(model)
    if layer_names:
        by_name = {type(m).__name__: type(m) for m in model.modules()}
        named = {by_name[name] for name in layer_names if name in by_name}
        if named:
            layer_classes = named

    if not layer_classes:
        raise ValueError(
            f"No transformer layer classes found for model {model.__class__.__name__}"
        )

    wrap_policy = partial(
        transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
    )

    import torch_xla.runtime as xr

    if fsdp_uses_spmd():
        from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
            SpmdFullyShardedDataParallel,
        )

        xr.use_spmd()

        n = get_xla_device_count()
        from torch_xla.distributed.spmd import Mesh

        mesh = Mesh(list(range(n)), (n,), ("fsdp",))

        model = SpmdFullyShardedDataParallel(
            model,
            mesh=mesh,
            shard_output=_spmd_shard_output,
            auto_wrap_policy=wrap_policy,
        )
    else:
        from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

        model = XlaFullyShardedDataParallel(
            model,
            auto_wrap_policy=wrap_policy,
            flatten_parameters=False,
            compute_dtype=torch.bfloat16,
        )

    return model


def unwrap_fsdp_model(model: nn.Module) -> nn.Module:
    """Unwrap an FSDP-wrapped model to the underlying module (best effort)."""
    if not is_torch_xla_available():
        return model

    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    if isinstance(model, XlaFullyShardedDataParallel):
        return model._fsdp_wrapped_module  # type: ignore[attr-defined]

    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel,
    )

    if isinstance(model, SpmdFullyShardedDataParallel):
        return model.module

    return model


def is_fsdp_model(model: nn.Module) -> bool:
    """Check if a model is wrapped with FSDP (either variant)."""
    if not is_torch_xla_available():
        return False

    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    if isinstance(model, XlaFullyShardedDataParallel):
        return True

    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel,
    )

    return isinstance(model, SpmdFullyShardedDataParallel)
