import torch
import torch.nn.functional as F


def _flatten_off_axis(square_matrix: torch.Tensor) -> torch.Tensor:
    mask = ~torch.eye(square_matrix.size(0), dtype=torch.bool, device=square_matrix.device)
    return square_matrix[mask]


def row_lengths(batch: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return torch.linalg.norm(batch, dim=dim)


def feature_spread(batch: torch.Tensor) -> torch.Tensor:
    centered = batch - batch.mean(dim=0, keepdim=True)
    return torch.sqrt(centered.var(dim=0) + 1e-4)


def normalized_feature_spread(batch: torch.Tensor) -> torch.Tensor:
    lengths = row_lengths(batch).unsqueeze(1)
    return feature_spread(batch / (lengths + 1e-6)).mean()


def spread_ratio(batch: torch.Tensor) -> torch.Tensor:
    return feature_spread(batch).mean() / (row_lengths(batch).mean() + 1e-6)


def _variance_guard(batch: torch.Tensor) -> torch.Tensor:
    return F.relu(1 - feature_spread(batch)).mean() / 2


def variance_guard_loss(
    batch_a: torch.Tensor,
    batch_b: torch.Tensor | None = None,
) -> torch.Tensor:
    if batch_b is None:
        return _variance_guard(batch_a)
    return _variance_guard(batch_a) + _variance_guard(batch_b)


def covariance_penalty(batch: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    centered = batch - batch.mean(dim=0, keepdim=True)
    denom = max(int(batch.size(0)) - 1, 1)
    covariance = centered.T @ centered / denom
    score = _flatten_off_axis(covariance).pow(2).sum()
    if normalize:
        score = score / batch.size(1)
    return score


def decorrelation_loss(
    batch_a: torch.Tensor,
    batch_b: torch.Tensor | None = None,
) -> torch.Tensor:
    if batch_b is None:
        return covariance_penalty(batch_a)
    return covariance_penalty(batch_a) + covariance_penalty(batch_b)


def normalized_covariance_penalty(batch: torch.Tensor) -> torch.Tensor:
    centered = batch - batch.mean(dim=0, keepdim=True)
    normalized = F.normalize(centered, p=2, dim=0)
    gram = normalized.T @ normalized
    return _flatten_off_axis(gram).pow(2).mean()


def covariance_rank_estimate(batch: torch.Tensor) -> torch.Tensor:
    centered = batch - batch.mean(dim=0, keepdim=True)
    denom = max(int(batch.size(0)) - 1, 1)
    covariance = centered.T @ centered / denom
    try:
        return torch.linalg.matrix_rank(covariance, hermitian=True).float()
    except RuntimeError:
        return torch.tensor(0.0)


def paired_mse_loss(batch_a: torch.Tensor, batch_b: torch.Tensor, *_unused) -> torch.Tensor:
    return F.mse_loss(batch_a, batch_b)
