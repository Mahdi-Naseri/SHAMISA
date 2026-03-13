import torch
import torch.nn as nn

try:
    from torch.nn.utils import parametrizations as nn_parametrizations
except ImportError:
    nn_parametrizations = None


class AnchorProjector(nn.Module):
    def __init__(self, anchor_cfg, feature_dim, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = int(feature_dim)
        self.anchor_count = int(anchor_cfg.n_prototypes)
        self.renormalize_rows = bool(anchor_cfg.prototype_norm)
        self.use_weight_norm = bool(anchor_cfg.weight_norm)

        projection = nn.Linear(self.feature_dim, self.anchor_count, bias=False)
        bound = (1.0 / self.feature_dim) ** 0.5
        nn.init.uniform_(projection.weight, -bound, bound)

        if self.use_weight_norm:
            if nn_parametrizations and hasattr(nn_parametrizations, "weight_norm"):
                projection = nn_parametrizations.weight_norm(
                    projection,
                    name="weight",
                    dim=0,
                )
            else:
                projection = nn.utils.weight_norm(projection, name="weight", dim=0)
        self.head = projection

    @property
    def weight(self):
        return self.head.weight

    def _refresh_rows(self) -> None:
        if not self.renormalize_rows:
            return
        with torch.no_grad():
            normalized = nn.functional.normalize(self.head.weight.data.clone())
            self.head.weight.data.copy_(normalized)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self._refresh_rows()
        return self.head(features)
