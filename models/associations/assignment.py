import math

import torch
import torch.nn.functional as F

from .losses import covariance_penalty, feature_spread, row_lengths
from .synthetic_links import SyntheticLinkBuilder


@torch.no_grad()
def _balanced_codes(logits: torch.Tensor, epsilon: float, rounds: int) -> torch.Tensor:
    codes = torch.exp(logits / epsilon).T
    sample_count = codes.shape[1]
    anchor_count = codes.shape[0]
    codes = codes / codes.sum()

    for _ in range(rounds):
        codes = codes / codes.sum(dim=1, keepdim=True)
        codes = codes / anchor_count
        codes = codes / codes.sum(dim=0, keepdim=True)
        codes = codes / sample_count

    return (codes * sample_count).T


def _rescale_alignment(raw_scores: torch.Tensor, mode: str, temperature: float) -> torch.Tensor:
    if mode == "min_max":
        shifted = raw_scores - raw_scores.min()
        scale = shifted.max()
        if scale != 0:
            shifted = shifted / scale
        return shifted
    if mode == "sigmoid":
        return torch.sigmoid(raw_scores / temperature)
    raise ValueError(f"Unsupported transport normalization '{mode}'")


def _retain_links(score_map: torch.Tensor, mode: str, threshold: float, topk_scale: int) -> tuple[torch.Tensor, torch.Tensor]:
    device = score_map.device

    if mode == "threshold":
        keep = score_map >= threshold
        keep_index = keep.nonzero(as_tuple=False).T
        return keep_index, score_map[keep]

    if mode == "topk_each":
        per_row = min(int(topk_scale), int(score_map.shape[1]))
        kept_values, kept_cols = torch.topk(score_map, per_row, dim=1, sorted=False)
        kept_rows = torch.arange(score_map.shape[0], device=device).repeat_interleave(per_row)
        return torch.stack((kept_rows, kept_cols.reshape(-1))), kept_values.reshape(-1)

    if mode == "topk_global":
        keep_count = min(int(topk_scale) * int(score_map.shape[0]), int(score_map.numel()))
        kept_values, flat_positions = torch.topk(score_map.reshape(-1), keep_count, sorted=False)
        kept_rows = flat_positions // score_map.shape[1]
        kept_cols = flat_positions % score_map.shape[1]
        return torch.stack((kept_rows, kept_cols)), kept_values

    raise ValueError(f"Unsupported transport sparsification '{mode}'")


def build_assignment_links(
    embeddings: torch.Tensor,
    anchor_projector,
    block_cfg,
    layout,
    distortion_payload: dict,
    total_nodes: int,
    dtype: torch.dtype,
    profile_name: str,
):
    source_vectors = F.normalize(embeddings, dim=-1) if block_cfg.source_norm else embeddings

    anchor_logits = anchor_projector(source_vectors)
    balanced_codes = _balanced_codes(anchor_logits.detach(), block_cfg.eps, block_cfg.sinkhorn_iters)
    softened_logits = anchor_logits / block_cfg.temp
    anchor_probs = F.softmax(softened_logits, dim=-1)
    anchor_log_probs = F.log_softmax(softened_logits, dim=-1)

    if block_cfg.g_alignment == "cos_s":
        dense_scores = anchor_probs @ anchor_probs.T
    elif block_cfg.g_alignment in {"ce", "ce_sym"}:
        cross_entropy_like = (anchor_log_probs[:, None, :] * anchor_probs[None, :, :]).mean(dim=-1)
        dense_scores = cross_entropy_like
        if block_cfg.g_alignment == "ce_sym":
            dense_scores = 0.5 * (dense_scores + dense_scores.T)
        dense_scores = _rescale_alignment(dense_scores, block_cfg.g_norm, block_cfg.g_norm_temp)
    else:
        raise ValueError(f"Unsupported transport alignment '{block_cfg.g_alignment}'")

    link_index, link_values = _retain_links(
        dense_scores,
        block_cfg.g_sparse,
        block_cfg.g_sparse_threshold,
        block_cfg.g_sparse_k,
    )
    link_map = torch.sparse_coo_tensor(
        link_index,
        link_values.to(dtype=dtype),
        (total_nodes, total_nodes),
    ).coalesce()

    structural_links = SyntheticLinkBuilder(layout, distortion_payload, profile_name).build(
        "structural",
        total_nodes,
        embeddings.device,
        dtype,
    )
    structural_index = structural_links.indices()

    self_term = torch.tensor(0.0, device=embeddings.device, dtype=dtype)
    if "self" in block_cfg.pq_alignment:
        self_term = (balanced_codes * anchor_log_probs).mean()

    paired_term = torch.tensor(0.0, device=embeddings.device, dtype=dtype)
    if "aug" in block_cfg.pq_alignment and structural_index.numel() > 0:
        paired_term = (
            balanced_codes[structural_index[0]] * anchor_log_probs[structural_index[1]]
        ).mean()

    alignment_penalty = -0.5 * block_cfg.pq_alignment_coeff * (self_term + paired_term)

    anchor_weights = anchor_projector.weight
    spread_penalty = torch.tensor(0.0, device=embeddings.device, dtype=dtype)
    if block_cfg.prots_var_coeff != 0:
        bank_view = anchor_weights
        if block_cfg.prots_var_norm:
            bank_view = bank_view / (row_lengths(bank_view).unsqueeze(1) + 1e-6)
            target_spread = 1 / math.sqrt(anchor_weights.shape[1])
        else:
            target_spread = 1.0
        spread_penalty = torch.relu(target_spread - feature_spread(bank_view)).mean()

    covariance_penalty_term = torch.tensor(0.0, device=embeddings.device, dtype=dtype)
    if block_cfg.prots_cov_coeff != 0:
        covariance_penalty_term = covariance_penalty(anchor_weights)

    bank_penalty = (
        block_cfg.prots_var_coeff * spread_penalty
        + block_cfg.prots_cov_coeff * covariance_penalty_term
    )
    return link_map, alignment_penalty, bank_penalty
