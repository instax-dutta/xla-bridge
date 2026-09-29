# xla-bridge

Eager inference and model surgery for Hugging Face Transformers on TPUs, via PyTorch/XLA.

## Why

The Hugging Face ecosystem covers TPU training (Trainer + XLA FSDP) and TPU serving (TGI/JetStream), but the library that wrapped the middle ground - `optimum-tpu` - is archived. What no surviving project provides is a maintained path for workflows that:

- run **eager PyTorch** on a TPU (custom forward passes, custom decode loops, hidden-state extraction) without Trainer or a serving stack,
- perform **model surgery** (LoRA adapter application, merge, clean re-export) on models hosted on XLA devices,
- run **single-process on multi-core TPU VMs** (Kaggle, Colab) where SPMD sharding must be active before the XLA client initializes.

`xla-bridge` fills that gap. It is a thin, dependency-free utility layer: you keep your own code, and call into the bridge for the TPU-hostile parts.

## What it gives you

| Module | What it does |
|---|---|
| `detection` | torch_xla presence check without initialization, TPU detection, core count from PJRT env vars |
| `environment` | Env-first setup: `XLA_USE_SPMD=1` must be set before any device access or multi-chip runs segfault in `PjRtComputationClient::ExecuteReplicated` (torch_xla 2.8) |
| `device` | XLA device resolution, synchronous `mark_step`, XLA cache cleanup |
| `fsdp` | FSDP wrapping over all physical cores via SPMD mesh for single-process hosts, automatic transformer layer-class discovery, FSDP-tolerant unwrap |
| `decode` | Static-shape sliding-window greedy decode: one compiled graph for the whole generation instead of a recompile per step |
| `merge` | XLA-safe LoRA merge: materialize adapter state, reload base on CPU, merge and unload so exports match base-model structure exactly |

Everything is opt-in at runtime: on a machine without `torch_xla`, every function either returns a safe default or raises a clear error - never crashes the import.

## Battle-tested failure modes (torch_xla 2.8, TPU v5e)

These are documented in docstrings next to the code that avoids them, so future changes do not silently reintroduce them:

1. **SPMD after client init segfaults.** Enabling SPMD (`xr.use_spmd()`) after the XLA client has started goes through a force-replication path that crashes in `PjRtComputationClient::ExecuteReplicated`. Environment variables must be set before the first device access.
2. **Lazy execution + eager shape reads.** With `mark_step(wait=False)`, downstream eager ops that inspect shapes hit `Check failed: data->tensor_data`. Bridge steps always wait.
3. **Dropout RNG in the graph exhausts HBM.** An `.eval()` call before every forward keeps the HLO identical across steps; otherwise XLA recompiles per step and each executable keeps its own memory plan (~2.4 GB for a 0.5B model on v5e).
4. **Dynamic shapes recompile per batch.** Tokenization must pad to a fixed `max_length`, and the decode loop must use a fixed-size buffer with a constant last-position index, or XLA bakes a new graph per shape/index.
5. **Retained device tensors exhaust HBM.** Hidden states and logits must be offloaded to CPU (and concatenated there) instead of accumulated on-device.
6. **Single-core runs must not use SPMD.** The SPMD virtual device breaks memory probing and accumulates deviceless tensor fetches; SPMD is only enabled for the single-process multi-core FSDP path.

## Quickstart

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from xla_bridge import (
    detect_tpu,
    setup_tpu_environment,
    get_xla_device,
    greedy_decode,
    tokenize_fixed_length,
)

if not detect_tpu():
    raise SystemExit("No TPU detected")

# Must happen before any device access (SPMD env flag)
setup_tpu_environment()

device = get_xla_device()

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

model.eval()  # HLO stability - see failure mode 3

# Fixed-width left-padded window. Make the window comfortably wider than the
# longest prompt: the loop keeps only the most recent `window` tokens, so a
# tight window drops early prompt tokens that `model.generate()` would retain.
window = 128
prompt = tokenizer.apply_chat_template(
    [{"role": "user", "content": "Hello, my name is"}],
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenize_fixed_length(tokenizer, [prompt], max_length=window).to(device)

ids, finished = greedy_decode(
    model,
    inputs["input_ids"],
    inputs["attention_mask"],
    max_new_tokens=64,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
generated = ids[:, -64:]
print(tokenizer.batch_decode(generated, skip_special_tokens=True))
```

`greedy_decode` re-runs the full window each step instead of using a KV cache: that
trade is what keeps every shape static, so XLA compiles the loop once. It is intended
for short, fixed-length generations where HBM - not FLOPs - is the binding constraint.

## Installation

```bash
# On a TPU VM (e.g. Kaggle TPU v5e-8, torch_xla 2.8):
pip install "xla-bridge[transformers,tpu]" -f https://storage.googleapis.com/libtpu-releases/index.html

# Anywhere else (for type checking / non-TPU import safety):
pip install "xla-bridge[dev]"
```

Tested pairing: **torch 2.8.0 + torch_xla 2.8.0 on TPU v5e-8**. torch_xla 2.9 has an unrelated `std::bad_alloc` in `from_pretrained` on v5e and is not yet supported.

**Status note on validation.** All four modules were verified end-to-end on real TPU hardware:

- **Kaggle TPU v5e-8** (torch 2.8.0 / torch_xla 2.8.0, 8 cores, single-process SPMD FSDP), 2026-09-24: detection/env setup, static-shape greedy decode (first decode 7.0s incl. compile, subsequent 0.5s), FSDP wrap + sharded decode, XLA-safe LoRA merge with clean re-export, plus heretic's own 10/10 smoke suite and a 1-trial end-to-end abliteration with export.
- **Google Colab TPU v5e-1** (torch 2.9.0 / torch_xla 2.9.0, single chip), 2026-09-29: all modules verified again on a different substrate, and `greedy_decode` was checked **token-for-token against `model.generate()`** - exact match over 8 generated tokens when the sliding window is wider than the prompt. This run also found and fixed a real bug: the prompt must be right-aligned in the buffer, otherwise the first token is predicted from unwritten columns (see `decode.py`).

torch_xla 2.9 works on Colab v5e-1; it failed with `std::bad_alloc` in `from_pretrained` on the Kaggle v5e-8 image, so 2.8.x remains the pairing for multi-chip v5e-8.

## Relationship to other projects

- **`optimum-tpu`** (archived): covered Trainer-based training and TGI serving. `xla-bridge` is not a fork; it serves the eager-inference/model-surgery niche that was left open.
- **`transformers` Trainer XLA FSDP**: training-only, Trainer-coupled. Bridge FSDP wrapping is usable with plain modules and your own loops.
- **`torch_xla`**: the bridge codifies workarounds on top of its primitives (SPMD mesh, FSDPv2, PJRT envs); it is not a replacement.

## License

Apache-2.0.
