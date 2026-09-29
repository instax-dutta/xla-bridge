# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Abhishek Dash

"""XLA-safe text generation and tokenization helpers.

XLA compiles a separate HLO graph per distinct tensor shape (and per distinct
Python constant baked into the trace). The helpers below keep every tensor
shape and every traced constant fixed across the whole generation so the
model compiles ONCE and the compiled executable is reused for every decode
step. Without this, torch_xla recompiles per step (~25s compile each) and
each fresh executable keeps its own TPU memory plan, exhausting HBM.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .device import mark_step

if TYPE_CHECKING:
    import torch


def tokenize_fixed_length(
    tokenizer: Any,
    texts: list[str] | str,
    max_length: int,
    **kwargs: Any,
) -> Any:
    """Tokenize with padding="max_length" so every batch has an identical shape.

    Plain padding pads each batch to its own longest sequence, so batches
    of prompts with different lengths produce different shapes and XLA
    recompiles per batch. On TPU, force a fixed max_length for ALL batches.

    Padding side is forced to LEFT. Decoder-only generation reads the logits
    at the final position, so right padding would feed pad/EOS tokens as if
    they were prompt content and shift the whole generation. Temporarily
    override the tokenizer's padding_side so callers do not have to.

    The original padding_side is restored before returning.
    """
    original_side = getattr(tokenizer, "padding_side", None)
    if original_side is not None:
        tokenizer.padding_side = "left"
    try:
        return tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
            **kwargs,
        )
    finally:
        if original_side is not None:
            tokenizer.padding_side = original_side


def greedy_decode(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Greedy generation via a static-shape sliding-window loop.

    Avoids model.generate(), whose Python control flow forces XLA to fall
    back to CPU. Uses a fixed-size pre-allocated buffer so tensor shapes
    never change across steps, and reads logits at a CONSTANT index (the
    last position): XLA bakes integer constants into the compiled HLO, so a
    varying index would produce a different graph per decode step.

    Args:
        model: The (already eval()-ed) language model.
        input_ids: (batch, prompt_len) token ids, on the XLA device.
        attention_mask: (batch, prompt_len) attention mask, on the device.
        max_new_tokens: Number of tokens to generate.
        eos_token_id: End-of-sequence id, or None to generate unconditionally.
        pad_token_id: Token emitted after a sequence has finished.

    Returns:
        (full_ids, finished): the (batch, prompt_len + max_new_tokens) id
        buffer and a (batch,) bool tensor marking which sequences hit EOS.
        The last `max_new_tokens` columns hold the generated tokens; earlier
        columns hold the tail of the prompt that has not yet slid out of the
        window.

    Note: the prompt is right-aligned inside the buffer so the loop always
    reads the newest token. This is a full re-forward per step (no KV cache),
    so long generations cost O(window) compute per token - intended for
    short fixed-length generations where HBM, not FLOPs, is the constraint.
    """
    import torch

    with torch.no_grad():
        return _greedy_decode_impl(
            model,
            input_ids,
            attention_mask,
            max_new_tokens,
            eos_token_id,
            pad_token_id,
        )


def _greedy_decode_impl(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | None,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    device = input_ids.device
    batch_size, prompt_len = input_ids.shape
    total_len = prompt_len + max_new_tokens

    # Pre-allocate fixed-size buffers. Shape NEVER changes.
    #
    # The prompt is placed so that it ENDS at the last column: the loop reads
    # logits at the final column, so that column must always hold the most
    # recent token. Left-aligning the prompt inside a wider buffer (the naive
    # layout) makes step 0 predict from unwritten zero-token columns and
    # silently changes the first generated token.
    full_ids = torch.zeros(batch_size, total_len, device=device, dtype=input_ids.dtype)
    full_ids[:, -prompt_len:] = input_ids
    full_mask = torch.zeros(
        batch_size, total_len, device=device, dtype=attention_mask.dtype
    )
    full_mask[:, -prompt_len:] = attention_mask

    finished = torch.zeros(batch_size, device=device, dtype=torch.bool)
    eos_tensor = (
        torch.tensor(eos_token_id, device=device, dtype=torch.long)
        if eos_token_id is not None
        else None
    )

    for _ in range(max_new_tokens):
        # Always pass the full fixed-size tensor - XLA sees the same shape.
        outputs = model(input_ids=full_ids, attention_mask=full_mask)

        # Read logits at the LAST position. This index must stay a Python
        # constant across steps (see module docstring).
        next_logits = outputs.logits[:, -1, :]
        next_token = next_logits.argmax(dim=-1)

        if eos_tensor is not None:
            next_token = torch.where(
                finished, torch.full_like(next_token, pad_token_id), next_token
            )

        # Shift the window one token left (constant-shape op) and write the
        # new token at the fixed last position, so every step's graph is
        # identical and the whole decode loop compiles once.
        full_ids[:, :-1] = full_ids[:, 1:]
        full_ids[:, -1] = next_token
        full_mask[:, :-1] = full_mask[:, 1:]
        full_mask[:, -1] = 1

        if eos_tensor is not None:
            finished = finished | (next_token == eos_tensor)

        # mark_step per step: keeps each execution small. A single unrolled
        # graph for all steps instead exhausts HBM via per-step logits
        # buffers and thrashes evictions.
        mark_step()

    mark_step()
    return full_ids, finished
