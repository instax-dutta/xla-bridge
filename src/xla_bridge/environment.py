# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""Environment setup for TPU runs.

CRITICAL ORDERING: torch_xla 2.8 initializes its XLA client lazily on the
first device access. When a single process drives multiple TPU chips, SPMD
must be active from the moment the client starts. Setting XLA_USE_SPMD=1 in
the environment before any device access makes the client initialize in SPMD
mode; calling xr.use_spmd() after client init goes through a force-replication
path that segfaults in PjRtComputationClient::ExecuteReplicated (a plain
sharded matmul reproduces it).

Consequence: ALWAYS call setup_tpu_environment() (or ensure_spmd()) before
touching any XLA device, model, or tensor.
"""

import os

from .detection import get_tpu_core_count_from_env, is_torch_xla_available


def setup_tpu_environment(enable_spmd: bool = False) -> None:
    """Set environment variables for optimal TPU behavior.

    Idempotent (setdefault) and safe on non-TPU hosts.

    Args:
        enable_spmd: Set XLA_USE_SPMD=1 for the single-process multi-core
            FSDP path. Only set this when you actually need SPMD sharding:
            single-core runs must stay in plain multi-device mode because the
            SPMD virtual device breaks memory probing and accumulates
            deviceless tensor fetches.
    """
    os.environ.setdefault("PJRT_DEVICE", "TPU")
    os.environ.setdefault("XLA_USE_BF16", "1")
    os.environ.setdefault("XLA_DOWNCAST_BF16", "1")
    if enable_spmd:
        os.environ.setdefault("XLA_USE_SPMD", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure_spmd_if_multichip(enable: bool = False) -> None:
    """Enable SPMD mode before the XLA client initializes, if needed.

    Safety net for callers that did not call setup_tpu_environment first.
    Only acts on single-process hosts with multiple physical cores. Safe to
    call repeatedly; a no-op on non-TPU hosts and single-core runs.
    """
    if not enable or not is_torch_xla_available():
        return
    try:
        os.environ.setdefault("XLA_USE_SPMD", "1")
        import torch_xla.runtime as xr

        if xr.is_spmd():
            return
        if xr.process_count() == 1 and get_tpu_core_count_from_env() > 1:
            xr.use_spmd()
    except Exception:
        pass
