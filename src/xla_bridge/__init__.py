# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""Eager inference and model surgery for Transformers on TPUs via PyTorch/XLA."""

from .decode import greedy_decode, tokenize_fixed_length
from .detection import (
    detect_tpu,
    get_tpu_core_count_from_env,
    is_torch_xla_available,
)
from .device import (
    empty_cache,
    get_xla_device,
    get_xla_device_count,
    mark_step,
)
from .environment import setup_tpu_environment

__all__ = [
    "detect_tpu",
    "empty_cache",
    "get_tpu_core_count_from_env",
    "get_xla_device",
    "get_xla_device_count",
    "greedy_decode",
    "is_torch_xla_available",
    "mark_step",
    "setup_tpu_environment",
    "tokenize_fixed_length",
]
