import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet50


class ResNet(nn.Module):
    """
    ResNet model with a projection head.

    Args:
        embedding_dim (int): embedding dimension of the projection head
        pretrained (bool): whether to use pretrained weights
        use_norm (bool): legacy flag to normalize both reps and embeddings
        use_norm_rep (bool): normalize representations
        use_norm_emb (bool): normalize embeddings
    """

    def __init__(
        self,
        embedding_dim: int,
        projector_out_dim: int | None = None,
        projector_hidden_dim: int | None = None,
        pretrained: bool = True,
        use_norm_rep: bool = True,
        use_norm_emb: bool = True,
        use_norm: bool | None = None,
    ):
        super(ResNet, self).__init__()

        self.pretrained = pretrained
        if use_norm is not None:
            use_norm_rep = bool(use_norm)
            use_norm_emb = bool(use_norm)
        self.use_norm_rep = use_norm_rep
        self.use_norm_emb = use_norm_emb
        self.embedding_dim = embedding_dim
        self.projector_out_dim = self._resolve_projector_dim(
            projector_out_dim, embedding_dim
        )

        if self.pretrained:
            weights = (
                torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            )  # V1 weights work better than V2
        else:
            weights = None
        self.model = resnet50(weights=weights)

        self.feat_dim = self.model.fc.in_features
        self.projector_hidden_dim = self._resolve_projector_dim(
            projector_hidden_dim, self.feat_dim
        )
        self.model = nn.Sequential(*list(self.model.children())[:-1])

        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.projector_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.projector_hidden_dim, self.projector_out_dim),
        )

    def forward(self, x):
        f = self.model(x)
        f = f.view(-1, self.feat_dim)

        if self.use_norm_rep:
            f = F.normalize(f, dim=1)

        g = self.projector(f)
        if self.use_norm_emb:
            return f, F.normalize(g, dim=1)
        else:
            return f, g

    @staticmethod
    def _resolve_projector_dim(projector_dim, fallback_dim: int) -> int:
        if projector_dim is None:
            return fallback_dim
        if isinstance(projector_dim, str):
            lowered = projector_dim.strip().lower()
            if lowered in {"none", "null", ""}:
                return fallback_dim
            try:
                projector_dim = float(projector_dim)
            except ValueError:
                return fallback_dim
        try:
            return int(projector_dim)
        except (TypeError, ValueError):
            return fallback_dim
