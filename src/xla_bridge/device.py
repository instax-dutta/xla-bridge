# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""XLA device resolution and execution control."""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

from .detection import get_tpu_core_count_from_env, is_torch_xla_available
from .environment import _ensure_spmd_if_multichip

if TYPE_CHECKING:
    import torch


def get_xla_device(core_id: int = 0, enable_spmd: bool = False) -> torch.device:
    """Get the XLA device for the given core ordinal.

    Args:
        core_id: Ordinal of the (virtual) device. Under SPMD there is a single
            virtual device, so leave this at 0.
        enable_spmd: Ensure SPMD is enabled before the client initializes.
            Must only be True for the single-process multi-core FSDP path.
    """
    if not is_torch_xla_available():
        raise RuntimeError("torch_xla not available")
    _ensure_spmd_if_multichip(enable=enable_spmd)

    import torch_xla.core.xla_model as xm

    return xm.xla_device(n=core_id)


def get_xla_device_count() -> int:
    """Number of physical TPU cores visible to this process.

    Falls back to PJRT env vars when runtime reports a single (SPMD virtual)
    device - torch_xla 2.8 quirk. Returns 0 when torch_xla is unavailable.
    """
    if not is_torch_xla_available():
        return 0
    try:
        import torch_xla.runtime as xr

        count = xr.global_device_count()
        if count > 1:
            return count
        from_env = get_tpu_core_count_from_env()
        return from_env if from_env > 1 else count
    except Exception:
        return 0


def mark_step(wait: bool = True) -> None:
    """Mark a step for XLA lazy execution. Call after each forward pass.

    wait=True is critical on torch_xla 2.8: with the default wait=False,
    execution is dispatched asynchronously and tensor device data is not
    attached yet when control returns. Downstream eager ops like torch.cat
    that inspect shapes then hit "Check failed: data->tensor_data" inside
    XLATensor::shape(). Waiting makes each step fully materialize.
    """
    if not is_torch_xla_available():
        return
    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step(wait=wait)
    except Exception:
        pass


def empty_cache() -> None:
    """Clear the XLA compilation cache and collect garbage.

    XLA caches compiled HLO graphs, which accumulate across repeated
    forwards and can exhaust system RAM during long runs. Call this between
    phases (e.g. after each optimization trial) on TPU runs.
    """
    gc.collect()
    try:
        import torch_xla.runtime as xr

        xr.reset_spmd()
    except Exception:
        pass
    try:
        import torch_xla.core.xla_model as xm

        xm.mark_step()
    except Exception:
        pass
    gc.collect()
