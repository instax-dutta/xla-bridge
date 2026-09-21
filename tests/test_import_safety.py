# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""Import-safety tests.

The library must import cleanly on ANY machine (no TPU, no torch_xla, even
without torch/transformers installed) - detection and environment helpers
are the one surface everyone touches first, so a crash there breaks the
"no-op off TPU" guarantee.
"""

import subprocess
import sys


def test_package_imports_without_torch_xla():
    # This test machine has torch but no torch_xla; importing must not raise.
    from xla_bridge import detection, environment

    assert detection.is_torch_xla_available() is False
    assert detection.detect_tpu() is False
    assert detection.get_tpu_core_count_from_env() == 1

    # Env setup is idempotent and safe anywhere.
    environment.setup_tpu_environment(enable_spmd=True)
    environment.setup_tpu_environment(enable_spmd=True)
    import os

    assert os.environ.get("PJRT_DEVICE") == "TPU"
    assert os.environ.get("XLA_USE_SPMD") == "1"


def test_imports_bare_interpreter():
    # Import every module in a bare interpreter with no optional deps forced.
    code = (
        "import xla_bridge, xla_bridge.detection, xla_bridge.environment, "
        "xla_bridge.device, xla_bridge.decode\n"
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
