import torch
import torch.nn
from contextlib import nullcontext
from torch_geometric.nn.models import MLP
from dotmap import DotMap
from models.resnet import ResNet
from models.associations import (
    AnchorProjector,
    AssociationPenalty,
    decorrelation_loss,
    paired_mse_loss,
    variance_guard_loss,
)

class Vicreg(torch.nn.Module):
    """
    Vicreg model class used for pre-training the encoder for IQA.

    Args:
        encoder_params (dict): encoder parameters with keys
            - embedding_dim (int): embedding dimension of the encoder projection head
            - pretrained (bool): whether to use pretrained weights for the encoder
        temperature (float): temperature for the loss function. Default: 0.1

    Returns:
        if training:
            loss (torch.Tensor): loss value
        if not training:
            q (torch.Tensor): image embeddings before the projection head (NxC)
            proj_q (torch.Tensor): image embeddings after the projection head (NxC)

    """

    def __init__(
        self,
        args: DotMap,
        projector_out_dim: int | None = None,
        projector_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.encoder_params = args.model.encoder
        self.args = args
        self.coeff = args.model.coeff
        projector_out_dim = self._resolve_projector_out_dim(projector_out_dim)
        projector_hidden_dim = self._resolve_projector_hidden_dim(
            projector_hidden_dim
        )
        self.encoder = ResNet(
            embedding_dim=self.encoder_params.embedding_dim,
            projector_out_dim=projector_out_dim,
            projector_hidden_dim=projector_hidden_dim,
            pretrained=self.encoder_params.pretrained,
            use_norm_rep=self.encoder_params.use_norm_rep,
            use_norm_emb=self.encoder_params.use_norm_emb,
        )
        if args.model.relations.active:
            self.association_penalty = AssociationPenalty(args.model.relations, self)
            self.invariance = self.association_penalty.compute_loss
            self._build_anchor_projector()
            self._build_branch_weighting()
        else:
            self.invariance = paired_mse_loss

    def forward(self, inp, batch_style_inp=False):
        if not self.training or not batch_style_inp:
            h, z = self.encoder(inp)
            return h, z

        batch = inp

        ref_imgs = batch["ref_imgs"]
        dist_imgs = batch["dist_imgs"]
        h_dim, w_dim = ref_imgs.shape[-2], ref_imgs.shape[-1]
        ref_imgs_compact = ref_imgs.view(-1, 3, h_dim, w_dim)
        dist_imgs_compact = dist_imgs.view(-1, 3, h_dim, w_dim)

        h_ref_compact, z_ref_compact = self.encoder(ref_imgs_compact)
        h_dist_compact, z_dist_compact = self.encoder(dist_imgs_compact)

        z_ref = z_ref_compact.view(
            ref_imgs.shape[:-3] + z_ref_compact.shape[-1:]
        )  # + concats two torch.Sizes
        h_ref = h_ref_compact.view(ref_imgs.shape[:-3] + h_ref_compact.shape[-1:])
        h_dist = h_dist_compact.view(dist_imgs.shape[:-3] + h_dist_compact.shape[-1:])
        z_dist = z_dist_compact.view(dist_imgs.shape[:-3] + z_dist_compact.shape[-1:])

        z_compact = torch.cat((z_ref_compact, z_dist_compact), dim=0)
        h_compact = torch.cat((h_ref_compact, h_dist_compact), dim=0)

        ref_half = int(z_ref_compact.shape[0] / 2)
        dist_half = int(z_dist_compact.shape[0] / 2)

        # Compute var/cov losses in fp32 to avoid AMP quantization plateaus.
        # This is especially important when std is low and close to the epsilon floor.
        autocast_off = (
            torch.cuda.amp.autocast(enabled=False)
            if z_compact.is_cuda
            else nullcontext()
        )
        with autocast_off:
            if self.args.model.single_view_var_cov:
                z_compact_fp32 = z_compact.float()
                var_loss = variance_guard_loss(z_compact_fp32)
                cov_loss = decorrelation_loss(z_compact_fp32)
            else:
                z1 = torch.cat(
                    (z_ref_compact[:ref_half], z_dist_compact[:dist_half]), dim=0
                )
                z2 = torch.cat(
                    (z_ref_compact[ref_half:], z_dist_compact[dist_half:]), dim=0
                )
                z1_fp32 = z1.float()
                z2_fp32 = z2.float()
                var_loss = variance_guard_loss(z1_fp32, z2_fp32)
                cov_loss = decorrelation_loss(z1_fp32, z2_fp32)

        if float(self.coeff.inv) == 0.0:
            inv_loss = torch.zeros((), device=var_loss.device, dtype=var_loss.dtype)
        else:
            inv_loss = self.invariance(
                z_ref,
                z_dist,
                h_ref,
                h_dist,
                z_ref_compact,
                z_dist_compact,
                h_ref_compact,
                h_dist_compact,
                z_compact,
                h_compact,
                batch["dist_comps"],
            )

        loss = (
            self.coeff.var * var_loss
            + self.coeff.inv * inv_loss.float()
            + self.coeff.cov * cov_loss
        )

        loss_terms_dict = {
            "var_loss": var_loss,
            "inv_loss": inv_loss,
            "cov_loss": cov_loss,
        }

        return loss, loss_terms_dict

    def _build_anchor_projector(self):
        self.anchor_projector = None
        relation_cfg = self.args.model.relations
        for branch in relation_cfg.branches.values():
            if not branch.active or branch.kind != "transport":
                continue
            source_dim = (
                self.encoder.feat_dim
                if branch.source == "rep"
                else self.encoder.projector[-1].out_features
            )
            self.anchor_projector = AnchorProjector(branch.transport, source_dim)
            break

    def _build_branch_weighting(self):
        self.branch_weight_mlp = None
        weighting_cfg = self.args.model.relations.weighting
        if not weighting_cfg.active:
            return

        branch_count = sum(
            1 for branch in self.args.model.relations.branches.values() if branch.active
        )
        input_channels = 2 * branch_count
        hidden_channels = input_channels * int(weighting_cfg.hidden_scale)
        self.branch_weight_mlp = torch.nn.Sequential(
            torch.nn.LayerNorm(input_channels),
            MLP(
                in_channels=input_channels,
                hidden_channels=hidden_channels,
                out_channels=branch_count,
                num_layers=int(weighting_cfg.num_layers),
                dropout=float(weighting_cfg.dropout),
                act=str(weighting_cfg.act),
                act_first=bool(weighting_cfg.act_first),
                norm=str(weighting_cfg.norm)
                if str(weighting_cfg.norm).lower() not in {"none", ""}
                else None,
                plain_last=bool(weighting_cfg.plain_last),
                bias=bool(weighting_cfg.bias),
            ),
        )

    def _resolve_projector_out_dim(self, projector_out_dim):  
        if projector_out_dim is not None:
            return projector_out_dim
        if not hasattr(self.args, "model"):
            return None
        model_cfg = getattr(self.args, "model", None)
        if model_cfg is None or not hasattr(model_cfg, "projector"):
            return None
        out_dim = getattr(model_cfg.projector, "out_dim", None)
        if isinstance(out_dim, str):
            lowered = out_dim.strip().lower()
            if lowered in {"none", "null", ""}:
                return None
            try:
                out_dim = float(out_dim)
            except ValueError:
                return None
        try:
            return int(out_dim)
        except (TypeError, ValueError):
            return None

    def _resolve_projector_hidden_dim(self, projector_hidden_dim):
        if projector_hidden_dim is not None:
            return projector_hidden_dim
        if not hasattr(self.args, "model"):
            return None
        model_cfg = getattr(self.args, "model", None)
        if model_cfg is None or not hasattr(model_cfg, "projector"):
            return None
        hidden_dim = getattr(model_cfg.projector, "hidden_dim", None)
        if isinstance(hidden_dim, str):
            lowered = hidden_dim.strip().lower()
            if lowered in {"none", "null", ""}:
                return None
            try:
                hidden_dim = float(hidden_dim)
            except ValueError:
                return None
        try:
            return int(hidden_dim)
        except (TypeError, ValueError):
            return None
