from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .assignment import build_assignment_links
from .indexing import SampleLayout
from .neighborhoods import build_neighbor_links
from .synthetic_links import SyntheticLinkBuilder


@dataclass
class BranchSpec:
    label: str
    cfg: object

    @property
    def mode(self) -> str:
        return str(self.cfg.kind)

    @property
    def source(self) -> str:
        return str(getattr(self.cfg, "source", "rep"))


class AssociationPenalty:
    def __init__(self, association_cfg, owner):
        self.association_cfg = association_cfg
        self.owner = owner
        self.branch_specs = [
            BranchSpec(label=name, cfg=association_cfg.branches[name])
            for name in association_cfg.branches
            if association_cfg.branches[name].active
        ]

    def _branch_weights(self, link_maps: list[torch.Tensor]) -> torch.Tensor | None:
        weighting_cfg = self.association_cfg.weighting
        if not weighting_cfg.active or getattr(self.owner, "branch_weight_mlp", None) is None:
            return None

        summary_fields = []
        device = link_maps[0].device if link_maps else torch.device("cpu")
        for link_map in link_maps:
            summary_fields.append(torch.tensor(float(link_map.values().numel()), device=device))
            summary_fields.append(link_map.values().sum().detach())
        stats_vector = torch.stack(summary_fields)
        if weighting_cfg.stop_grad:
            stats_vector = stats_vector.detach()
        logits = self.owner.branch_weight_mlp(stats_vector.unsqueeze(0)).squeeze(0)
        weights = F.softmax(logits, dim=-1)
        return weights.detach() if weighting_cfg.stop_grad else weights

    def _scale_link_maps(self, link_maps: list[torch.Tensor]) -> list[torch.Tensor]:
        weights = self._branch_weights(link_maps)
        if weights is None:
            return link_maps

        scaled_maps = []
        for link_map, weight in zip(link_maps, weights):
            scaled_maps.append(
                torch.sparse_coo_tensor(
                    link_map.indices(),
                    link_map.values() * weight,
                    link_map.shape,
                ).coalesce()
            )
        return scaled_maps

    def _combine_link_maps(self, named_links: dict[str, torch.Tensor]) -> torch.Tensor:
        knn_label = next(
            (spec.label for spec in self.branch_specs if spec.mode == "knn"),
            None,
        )
        knn_links = named_links.get(knn_label) if knn_label is not None else None
        merged_links = None

        for spec in self.branch_specs:
            link_map = named_links[spec.label]
            if spec.mode == "knn":
                continue
            if getattr(spec.cfg, "multiply_by_knn", False) and knn_links is not None:
                link_map = (link_map * knn_links).coalesce()
            merged_links = link_map if merged_links is None else (merged_links + link_map).coalesce()

        if merged_links is None:
            merged_links = knn_links
        elif self.association_cfg.include_knn_residual and knn_links is not None:
            merged_links = (merged_links + knn_links).coalesce()

        return merged_links.coalesce()

    def _prepare_edges(self, link_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        edge_index = link_map.indices()
        edge_values = link_map.values()
        if not self.association_cfg.symmetric:
            return edge_index, edge_values
        keep = edge_index[0] < edge_index[1]
        return edge_index[:, keep], edge_values[keep]

    def _regularize_edges(self, edge_values: torch.Tensor) -> torch.Tensor:
        regularizer_cfg = self.association_cfg.regularizer
        if not regularizer_cfg.active:
            return torch.tensor(0.0, device=edge_values.device)
        if regularizer_cfg.mode == "neg_sum_abs":
            penalty = -torch.abs(edge_values).sum()
        else:
            penalty = -torch.square(edge_values).sum()
        return penalty * regularizer_cfg.coeff

    def _pairwise_cost(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_values: torch.Tensor,
    ) -> torch.Tensor:
        if edge_values.numel() == 0:
            return torch.tensor(0.0, device=embeddings.device)

        if self.association_cfg.pair_metric == "euc":
            pair_distances = (embeddings[edge_index[0]] - embeddings[edge_index[1]]).pow(2).sum(dim=1)
            if not self.association_cfg.soft_afgrl:
                return pair_distances.sum() / (embeddings.size(0) * embeddings.size(1))
            return pair_distances.inner(edge_values) / (embeddings.size(0) * embeddings.size(1))

        if self.association_cfg.pair_metric == "cos":
            normalized = F.normalize(embeddings, dim=-1)
            cosine_gap = 1 - (normalized[edge_index[0]] * normalized[edge_index[1]]).sum(dim=-1)
            return cosine_gap.inner(edge_values) / embeddings.size(0)

        raise ValueError(f"Unsupported pair metric '{self.association_cfg.pair_metric}'")

    def _select_features(
        self,
        spec: BranchSpec,
        embeddings: torch.Tensor,
        representations: torch.Tensor,
    ) -> torch.Tensor:
        if spec.source == "rep":
            return representations
        if spec.source == "emb":
            return embeddings
        raise ValueError(f"Unsupported feature source '{spec.source}'")

    def _build_branch_links(
        self,
        spec: BranchSpec,
        layout: SampleLayout,
        distortion_payload: dict,
        embeddings: torch.Tensor,
        representations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_nodes = int(embeddings.size(0))
        dtype = embeddings.dtype
        device = embeddings.device
        synthetic_builder = SyntheticLinkBuilder(
            layout,
            distortion_payload,
            self.association_cfg.distortion_curve,
        )

        if spec.mode == "metadata":
            link_map = synthetic_builder.build(
                spec.cfg.relation,
                total_nodes,
                device,
                dtype,
                getattr(spec.cfg, "topk", None),
            )
            zero = torch.tensor(0.0, device=device, dtype=dtype)
            return link_map, zero, zero

        source_vectors = self._select_features(spec, embeddings, representations)

        if spec.mode == "knn":
            link_map = build_neighbor_links(
                source_vectors,
                neighbor_count=spec.cfg.k,
                metric_name=spec.cfg.metric,
                include_self=spec.cfg.include_self,
                symmetric=spec.cfg.symmetric,
                soft_entries=spec.cfg.soft_entries,
            )
            zero = torch.tensor(0.0, device=device, dtype=dtype)
            return link_map, zero, zero

        if spec.mode == "transport":
            return build_assignment_links(
                source_vectors,
                self.owner.anchor_projector,
                spec.cfg.transport,
                layout,
                distortion_payload,
                total_nodes,
                dtype,
                self.association_cfg.distortion_curve,
            )

        raise ValueError(f"Unsupported branch kind '{spec.mode}'")

    def compute_loss(
        self,
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
        dist_comps,
    ):
        layout = SampleLayout(
            z_ref,
            z_dist,
            h_ref,
            h_dist,
            z_ref_compact,
            z_dist_compact,
            h_ref_compact,
            h_dist_compact,
        )

        named_links = {}
        alignment_penalty = torch.tensor(0.0, device=z_compact.device, dtype=z_compact.dtype)
        bank_penalty = torch.tensor(0.0, device=z_compact.device, dtype=z_compact.dtype)

        for spec in self.branch_specs:
            link_map, branch_alignment, branch_bank_penalty = self._build_branch_links(
                spec,
                layout,
                dist_comps,
                z_compact,
                h_compact,
            )
            if float(spec.cfg.coeff) != 1.0:
                link_map = torch.sparse_coo_tensor(
                    link_map.indices(),
                    link_map.values() * float(spec.cfg.coeff),
                    link_map.shape,
                ).coalesce()
            named_links[spec.label] = link_map
            alignment_penalty = alignment_penalty + branch_alignment
            bank_penalty = bank_penalty + branch_bank_penalty

        ordered_links = [named_links[spec.label] for spec in self.branch_specs]
        weighted_links = self._scale_link_maps(ordered_links)
        named_links = {
            spec.label: link_map
            for spec, link_map in zip(self.branch_specs, weighted_links)
        }
        merged_links = self._combine_link_maps(named_links)

        edge_index, edge_values = self._prepare_edges(merged_links)
        pair_cost = self._pairwise_cost(z_compact, edge_index, edge_values)
        edge_regularizer = self._regularize_edges(edge_values)

        total = pair_cost + alignment_penalty + bank_penalty + edge_regularizer
        if self.association_cfg.symmetric:
            total = total * 2
        return total
