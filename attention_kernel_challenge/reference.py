from __future__ import annotations

from typing import Tuple

import torch

from .cases import ceil_div
from .spec import BLOCK_SIZE, HEAD_DIM, SCORE_SCALE


@torch.no_grad()
def reference_block_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, t_max, head_dim = q.shape
    if head_dim != HEAD_DIM:
        raise ValueError(f"Expected head_dim={HEAD_DIM}, got {head_dim}")
    if t_max % BLOCK_SIZE != 0:
        raise ValueError(f"Expected t_max to be a multiple of {BLOCK_SIZE}, got {t_max}")

    q_blocks = row_ptr.shape[-1] - 1
    batch_heads = batch_size * num_heads
    device = q.device

    q_blocks_tensor = _as_block_view(q, batch_size, num_heads, q_blocks, head_dim)
    k_blocks_tensor = _as_block_view(k, batch_size, num_heads, q_blocks, head_dim)
    v_blocks_tensor = _as_block_view(v, batch_size, num_heads, q_blocks, head_dim)

    row_ptr_flat = row_ptr.reshape(batch_heads, q_blocks + 1).to(torch.int64)
    col_idx_flat = col_idx.reshape(batch_heads, -1)
    seq_lens_flat = seq_lens[:, None].expand(batch_size, num_heads).reshape(batch_heads).to(torch.int64)
    valid_blocks = _ceil_div(seq_lens_flat, BLOCK_SIZE)
    token_offsets = torch.arange(BLOCK_SIZE, device=device, dtype=torch.int64)

    output = torch.zeros((batch_heads, q_blocks, BLOCK_SIZE, head_dim), device=device, dtype=torch.float32)
    lse = torch.full((batch_heads, q_blocks, BLOCK_SIZE), -torch.inf, device=device, dtype=torch.float32)

    max_valid_block = int(valid_blocks.max().item())
    for q_block in range(max_valid_block):
        q_start = q_block * BLOCK_SIZE
        active_batch_heads = torch.nonzero(seq_lens_flat > q_start, as_tuple=False).squeeze(1)
        if active_batch_heads.numel() == 0:
            continue

        row_start = row_ptr_flat[active_batch_heads, q_block]
        row_end = row_ptr_flat[active_batch_heads, q_block + 1]
        degrees = row_end - row_start

        for degree in torch.unique(degrees, sorted=True).tolist():
            if degree <= 0:
                continue

            degree_mask = degrees == degree
            degree_batch_heads = active_batch_heads[degree_mask]
            degree_row_start = row_start[degree_mask]
            degree_offsets = torch.arange(degree, device=device, dtype=torch.int64)
            gather_idx = degree_row_start[:, None] + degree_offsets[None, :]
            bounded_idx = gather_idx.clamp(max=col_idx_flat.shape[1] - 1)
            selected_col_idx = col_idx_flat.index_select(0, degree_batch_heads)
            k_block_idx = selected_col_idx.gather(1, bounded_idx)
            valid_k_block = (
                (k_block_idx >= 0)
                & (k_block_idx <= q_block)
                & (k_block_idx < valid_blocks[degree_batch_heads, None])
            )
            safe_k_block_idx = torch.where(valid_k_block, k_block_idx, torch.zeros_like(k_block_idx)).to(
                torch.int64
            )

            selected_q = q_blocks_tensor.index_select(0, degree_batch_heads)
            selected_k = k_blocks_tensor.index_select(0, degree_batch_heads)
            selected_v = v_blocks_tensor.index_select(0, degree_batch_heads)
            q_chunk = selected_q[:, q_block].to(torch.float32)
            q_token_idx = q_start + token_offsets[None, :]
            query_valid = q_token_idx < seq_lens_flat[degree_batch_heads, None]

            running_max = torch.full(
                (degree_batch_heads.numel(), BLOCK_SIZE),
                -torch.inf,
                device=device,
                dtype=torch.float32,
            )
            running_sum = torch.zeros(
                (degree_batch_heads.numel(), BLOCK_SIZE),
                device=device,
                dtype=torch.float32,
            )
            running_out = torch.zeros(
                (degree_batch_heads.numel(), BLOCK_SIZE, head_dim),
                device=device,
                dtype=torch.float32,
            )

            for slot in range(degree):
                slot_k_block_idx = safe_k_block_idx[:, slot]
                slot_valid_k_block = valid_k_block[:, slot]
                gather_block = slot_k_block_idx[:, None, None, None].expand(-1, 1, BLOCK_SIZE, head_dim)
                k_chunk = torch.gather(selected_k, 1, gather_block).squeeze(1).to(torch.float32)
                v_chunk = torch.gather(selected_v, 1, gather_block).squeeze(1).to(torch.float32)

                scores = torch.matmul(q_chunk, k_chunk.transpose(1, 2)) * SCORE_SCALE
                k_token_idx = slot_k_block_idx[:, None] * BLOCK_SIZE + token_offsets[None, :]
                allow = (
                    slot_valid_k_block[:, None, None]
                    & query_valid[:, :, None]
                    & (k_token_idx[:, None, :] < seq_lens_flat[degree_batch_heads, None, None])
                )
                same_block = slot_k_block_idx == q_block
                if torch.any(same_block):
                    allow &= (~same_block[:, None, None]) | (k_token_idx[:, None, :] <= q_token_idx[:, :, None])

                scores = scores.masked_fill(~allow, -torch.inf)
                block_max = scores.max(dim=-1).values
                new_max = torch.maximum(running_max, block_max)
                prev_scale = torch.where(
                    torch.isfinite(running_max),
                    torch.exp(running_max - new_max),
                    torch.zeros_like(new_max),
                )
                finite_scores = torch.isfinite(scores)
                weights = torch.where(
                    finite_scores,
                    torch.exp(scores - new_max[:, :, None]),
                    torch.zeros_like(scores),
                )

                running_out = running_out * prev_scale[:, :, None] + torch.matmul(weights, v_chunk)
                running_sum = running_sum * prev_scale + weights.sum(dim=-1)
                running_max = torch.where(torch.isfinite(block_max), new_max, running_max)

            finite_rows = running_sum > 0
            if not torch.any(finite_rows):
                continue

            block_output = output[degree_batch_heads, q_block]
            block_output[finite_rows] = running_out[finite_rows] / running_sum[finite_rows].unsqueeze(1)
            output[degree_batch_heads, q_block] = block_output

            block_lse = lse[degree_batch_heads, q_block]
            block_lse[finite_rows] = running_max[finite_rows] + torch.log(running_sum[finite_rows])
            lse[degree_batch_heads, q_block] = block_lse

    return output.reshape(batch_size, num_heads, t_max, head_dim).to(torch.bfloat16), lse.reshape(
        batch_size, num_heads, t_max
    )


def dense_reference_block_sparse_attn_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_lens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, t_max, head_dim = q.shape
    output = torch.zeros((batch_size, num_heads, t_max, head_dim), device=q.device, dtype=torch.float32)
    lse = torch.full((batch_size, num_heads, t_max), -torch.inf, device=q.device, dtype=torch.float32)

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)
    v_f = v.to(torch.float32)

    for batch_index in range(batch_size):
        seq_len = int(seq_lens[batch_index].item())
        if seq_len <= 0:
            continue
        dense_mask = _dense_token_mask(row_ptr[batch_index], col_idx[batch_index], seq_len, t_max, q.device)

        for head_index in range(num_heads):
            q_chunk = q_f[batch_index, head_index, :seq_len]
            k_chunk = k_f[batch_index, head_index, :seq_len]
            v_chunk = v_f[batch_index, head_index, :seq_len]

            scores = torch.matmul(q_chunk, k_chunk.transpose(0, 1)) * SCORE_SCALE
            scores = scores.masked_fill(~dense_mask[head_index, :seq_len, :seq_len], -torch.inf)

            max_scores = torch.max(scores, dim=-1).values
            valid_rows = torch.isfinite(max_scores)

            if not torch.any(valid_rows):
                continue

            probs = torch.zeros_like(scores)
            probs[valid_rows] = torch.softmax(scores[valid_rows], dim=-1)
            output[batch_index, head_index, :seq_len] = torch.matmul(probs, v_chunk)
            lse[batch_index, head_index, :seq_len][valid_rows] = torch.logsumexp(scores[valid_rows], dim=-1)

    return output.to(torch.bfloat16), lse


def _accumulate_block(
    scores: torch.Tensor,
    v_chunk: torch.Tensor,
    running_max: torch.Tensor,
    running_sum: torch.Tensor,
    running_out: torch.Tensor,
) -> None:
    for row_index in range(scores.shape[0]):
        row_scores = scores[row_index]
        finite_mask = torch.isfinite(row_scores)
        if not torch.any(finite_mask):
            continue

        current_max = running_max[row_index]
        block_max = torch.max(row_scores[finite_mask])

        if torch.isfinite(current_max):
            new_max = torch.maximum(current_max, block_max)
            prev_scale = torch.exp(current_max - new_max)
        else:
            new_max = block_max
            prev_scale = torch.tensor(0.0, device=scores.device, dtype=torch.float32)

        weights = torch.exp(row_scores[finite_mask] - new_max)
        running_out[row_index] = (
            running_out[row_index] * prev_scale
            + torch.matmul(weights, v_chunk[finite_mask].to(torch.float32))
        )
        running_sum[row_index] = running_sum[row_index] * prev_scale + torch.sum(weights)
        running_max[row_index] = new_max


def _dense_token_mask(
    row_ptr: torch.Tensor,
    col_idx: torch.Tensor,
    seq_len: int,
    t_max: int,
    device: torch.device,
) -> torch.Tensor:
    num_heads = row_ptr.shape[0]
    q_blocks = row_ptr.shape[-1] - 1
    mask = torch.zeros((num_heads, t_max, t_max), dtype=torch.bool, device=device)
    valid_blocks = min(q_blocks, ceil_div(seq_len, BLOCK_SIZE))

    for head_index in range(num_heads):
        for q_block in range(valid_blocks):
            q_start = q_block * BLOCK_SIZE
            q_end = min(q_start + BLOCK_SIZE, seq_len)
            row_start = int(row_ptr[head_index, q_block].item())
            row_end = int(row_ptr[head_index, q_block + 1].item())
            for offset in range(row_start, row_end):
                k_block = int(col_idx[head_index, offset].item())
                if k_block < 0 or k_block > q_block or k_block >= valid_blocks:
                    continue
                k_start = k_block * BLOCK_SIZE
                k_end = min(k_start + BLOCK_SIZE, seq_len)
                mask[head_index, q_start:q_end, k_start:k_end] = True
                if q_block == k_block:
                    q_indices = torch.arange(q_start, q_end, device=device)
                    k_indices = torch.arange(k_start, k_end, device=device)
                    causal_mask = k_indices.unsqueeze(0) <= q_indices.unsqueeze(1)
                    mask[head_index, q_start:q_end, k_start:k_end] = causal_mask

    return mask


def _as_block_view(
    x: torch.Tensor,
    batch_size: int,
    num_heads: int,
    q_blocks: int,
    head_dim: int,
) -> torch.Tensor:
    return x.reshape(batch_size, num_heads, q_blocks, BLOCK_SIZE, head_dim).reshape(
        batch_size * num_heads, q_blocks, BLOCK_SIZE, head_dim
    )


def _ceil_div(values: torch.Tensor, divisor: int) -> torch.Tensor:
    return (values + divisor - 1) // divisor
