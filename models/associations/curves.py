import torch


def _scaled_gap(level_gap: torch.Tensor) -> torch.Tensor:
    return level_gap.float() / 4.0


def _exp_decay(level_gap: torch.Tensor) -> torch.Tensor:
    return torch.exp(-3.0 * level_gap)


def _reciprocal_decay(level_gap: torch.Tensor) -> torch.Tensor:
    return torch.reciprocal(1.0 + level_gap)


_PROFILES = {
    "exponential": _exp_decay,
    "original": _reciprocal_decay,
}


def severity_to_strength(level_gap: torch.Tensor, profile_name: str) -> torch.Tensor:
    try:
        profile = _PROFILES[profile_name]
    except KeyError as exc:
        choices = ", ".join(sorted(_PROFILES))
        raise ValueError(
            f"Unsupported distortion_curve '{profile_name}'. Available: {choices}"
        ) from exc
    return profile(_scaled_gap(level_gap))
