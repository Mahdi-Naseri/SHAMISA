"""
Utilities for keeping compatibility with deprecated torch.cuda.amp helpers.
"""

from functools import wraps


def patch_cuda_amp_custom_autocast(device_type: str = "cuda") -> None:
    """
    Wrap torch.cuda.amp custom_fwd/custom_bwd so they forward to the new torch.amp
    versions with an explicit device_type. Safe to call multiple times.
    """
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except Exception:  # pragma: no cover - torch might not be installed
        return

    torch_amp = getattr(torch, "amp", None)
    cuda_module = getattr(torch, "cuda", None)
    cuda_amp = getattr(cuda_module, "amp", None) if cuda_module else None

    if not (torch_amp and cuda_amp):
        return

    torch_custom_fwd = getattr(torch_amp, "custom_fwd", None)
    torch_custom_bwd = getattr(torch_amp, "custom_bwd", None)
    cuda_custom_fwd = getattr(cuda_amp, "custom_fwd", None)
    cuda_custom_bwd = getattr(cuda_amp, "custom_bwd", None)

    if torch_custom_fwd and cuda_custom_fwd and not getattr(
        cuda_custom_fwd, "_shamisa_device_patch", False
    ):

        @wraps(torch_custom_fwd)
        def wrapped_custom_fwd(*args, **kwargs):
            kwargs.setdefault("device_type", device_type)
            return torch_custom_fwd(*args, **kwargs)

        wrapped_custom_fwd._shamisa_device_patch = True  # type: ignore[attr-defined]
        cuda_amp.custom_fwd = wrapped_custom_fwd

    if torch_custom_bwd and cuda_custom_bwd and not getattr(
        cuda_custom_bwd, "_shamisa_device_patch", False
    ):

        @wraps(torch_custom_bwd)
        def wrapped_custom_bwd(*args, **kwargs):
            kwargs.setdefault("device_type", device_type)
            return torch_custom_bwd(*args, **kwargs)

        wrapped_custom_bwd._shamisa_device_patch = True  # type: ignore[attr-defined]
        cuda_amp.custom_bwd = wrapped_custom_bwd
