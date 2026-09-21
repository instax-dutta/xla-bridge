# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""TPU / torch_xla detection.

Detection is deliberately non-initializing: checking for torch_xla must never
trigger the XLA client, because SPMD and other env vars have to be in place
BEFORE the client starts (see environment.py).
"""

import os


def is_torch_xla_available() -> bool:
    """Check if torch_xla is importable without initializing it."""
    try:
        import torch_xla  # noqa: F401

        return True
    except ImportError:
        return False


def detect_tpu() -> bool:
    """Detect if running on TPU (PyTorch/XLA).

    Checks PJRT_DEVICE first (set by TPU VMs), then falls back to asking the
    XLA device for its hardware type. Safe to call on any machine.
    """
    if not is_torch_xla_available():
        return False
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    try:
        import torch_xla.core.xla_model as xm

        return xm.xla_device_hw(xm.xla_device()) == "TPU"
    except Exception:
        return False


def get_tpu_core_count_from_env() -> int:
    """Total TPU cores visible to this host, from PJRT environment variables.

    torch_xla 2.8 in single-process SPMD mode reports one virtual device, so
    the physical core count must be derived from the environment:
    product(TPU_CHIPS_PER_HOST_BOUNDS) * product(TPU_HOST_BOUNDS).

    Returns 1 when the variables are absent or unparsable.
    """
    import math
    from contextlib import suppress

    total = 1
    for var in ("TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS"):
        val = os.environ.get(var)
        if val:
            with suppress(ValueError):
                total *= math.prod(int(dim) for dim in val.split(","))
    return total
