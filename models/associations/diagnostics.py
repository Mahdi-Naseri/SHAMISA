import torch

from .losses import (
    covariance_penalty,
    covariance_rank_estimate,
    feature_spread,
    normalized_covariance_penalty,
    normalized_feature_spread,
    paired_mse_loss,
    row_lengths,
    spread_ratio,
)


def _blend(primary: torch.Tensor, secondary: torch.Tensor | None, symmetric: bool) -> torch.Tensor:
    pieces = [primary]
    if symmetric and secondary is not None:
        pieces.append(secondary)
    return torch.stack(pieces)


def covariance_energy_metric(primary: torch.Tensor, secondary: torch.Tensor, symmetric: bool = False) -> float:
    return _blend(
        covariance_penalty(primary),
        covariance_penalty(secondary),
        symmetric,
    ).mean().item()


def normalized_covariance_metric(primary: torch.Tensor, secondary: torch.Tensor, symmetric: bool = False) -> float:
    return _blend(
        normalized_covariance_penalty(primary),
        normalized_covariance_penalty(secondary),
        symmetric,
    ).mean().item()


def mse_agreement_metric(primary: torch.Tensor, secondary: torch.Tensor, **_kwargs) -> float:
    return paired_mse_loss(primary, secondary).item()


def feature_spread_metric(primary: torch.Tensor, secondary: torch.Tensor | None = None, symmetric: bool = False) -> float:
    merged = _blend(feature_spread(primary), feature_spread(secondary) if secondary is not None else None, symmetric)
    return merged.mean(dim=1).mean().item()


def feature_norm_metric(primary: torch.Tensor, secondary: torch.Tensor | None = None, symmetric: bool = False) -> float:
    merged = _blend(row_lengths(primary), row_lengths(secondary) if secondary is not None else None, symmetric)
    return merged.mean(dim=1).mean().item()


def covariance_rank_metric(primary: torch.Tensor, secondary: torch.Tensor | None = None, symmetric: bool = False) -> float:
    merged = _blend(
        covariance_rank_estimate(primary.cpu()),
        covariance_rank_estimate(secondary.cpu()) if secondary is not None else None,
        symmetric,
    )
    return merged.mean().item()


def normalized_spread_metric(primary: torch.Tensor, secondary: torch.Tensor | None = None, symmetric: bool = False) -> float:
    return _blend(
        normalized_feature_spread(primary),
        normalized_feature_spread(secondary) if secondary is not None else None,
        symmetric,
    ).mean().item()


def spread_ratio_metric(primary: torch.Tensor, secondary: torch.Tensor, symmetric: bool = False) -> float:
    return _blend(spread_ratio(primary), spread_ratio(secondary), symmetric).mean().item()
