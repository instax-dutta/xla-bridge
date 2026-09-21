# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""XLA-safe LoRA merge and export for PEFT models.

Merging on-device on TPU is unreliable in torch_xla 2.8: cloning/copying XLA
lazy tensors after adapter updates fails with OSError EPERM (errno 1) when
HBM is near capacity, and FSDP wrapping pollutes parameter names with
_orig_module. prefixes. The safe path everywhere - including CPU/CUDA - is:

1. Materialize the adapter state (LoRA A/B weights) as plain CPU tensors.
2. Reload the base model on CPU in the target dtype.
3. Re-apply the PEFT config, copy the adapter weights in, merge_and_unload.
4. Export. The result matches the base model structure exactly.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch import nn


def _first_parameter(model: nn.Module) -> torch.Tensor | None:
    return next(iter(model.parameters()), None)


def _model_device(model: nn.Module) -> torch.device:
    device = getattr(model, "device", None)
    if isinstance(device, torch.device):
        return device
    param = _first_parameter(model)
    return param.device if param is not None else torch.device("cpu")


def _model_dtype(model: nn.Module) -> torch.dtype:
    dtype = getattr(model, "dtype", None)
    if isinstance(dtype, torch.dtype):
        return dtype
    param = _first_parameter(model)
    return param.dtype if param is not None else torch.float32


def materialize_adapter_state(
    model: nn.Module,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Clone all LoRA adapter parameters to a device (default CPU).

    Call this while the PEFT model is still intact, BEFORE moving it to CPU
    or unloading anything. Returned keys are fully qualified parameter names,
    directly compatible with apply_adapter_state().
    """
    adapter_state: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            adapter_state[name] = param.data.clone().to(device)
    return adapter_state


def apply_adapter_state(
    model: nn.Module,
    adapter_state: dict[str, torch.Tensor],
) -> None:
    """Copy previously materialized LoRA weights into a (freshly wrapped) model."""
    for name, param in model.named_parameters():
        if name in adapter_state:
            param.data = adapter_state[name].to(param.device)


def merge_peft_model_on_cpu(
    model: Any,
    reload_base: Callable[[], Any],
) -> Any:
    """Merge LoRA adapters into a CPU reload of the base model.

    Works for every host type (CPU, CUDA, TPU, FSDP-wrapped models); on TPU
    it is not merely safe but mandatory (see module docstring). On TPU the
    model is moved to CPU first: cloning XLA lazy tensors fails when HBM is
    near capacity, and CPU merging avoids TPU device issues entirely.

    Args:
        model: The current PEFT model (may be FSDP-wrapped or on XLA device).
            A `peft_config` attribute is required to re-wrap the fresh base.
        reload_base: Callable that rebuilds a fresh base model on CPU with
            the desired dtype/revision/quantization, e.g. `lambda:
            AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16,
            device_map="cpu")`.

    Returns:
        The merged (adapter-free) base model, matching base structure exactly.
    """
    if str(_model_device(model)).startswith("xla"):
        model = model.to("cpu")

    adapter_state = materialize_adapter_state(model)
    peft_config = model.peft_config

    base_model = reload_base()
    from peft import get_peft_model

    peft_model = get_peft_model(base_model, peft_config)
    apply_adapter_state(peft_model, adapter_state)

    return peft_model.merge_and_unload()


def save_merged_model(
    merged_model: nn.Module,
    save_directory: str,
    tokenizer: Any = None,
    processor: Any = None,
    max_shard_size: str | int = "5GB",
) -> None:
    """Save a merged model plus optional tokenizer/processor.

    On any host this uses the standard save_pretrained path, which produces
    safetensors shards. (Direct on-device state_dict dumps such as
    xm.save(pytorch_model.bin) produce a single pickle that HF repos cannot
    shard or load with safetensors; prefer a CPU-side save.)
    """
    import os

    os.makedirs(save_directory, exist_ok=True)
    merged_model.save_pretrained(save_directory, max_shard_size=max_shard_size)
    if tokenizer is not None:
        tokenizer.save_pretrained(save_directory)
    if processor is not None:
        processor.save_pretrained(save_directory)


__all__ = [
    "apply_adapter_state",
    "materialize_adapter_state",
    "merge_peft_model_on_cpu",
    "save_merged_model",
]
