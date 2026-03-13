import torch

from .curves import severity_to_strength
from .indexing import SampleLayout


class SyntheticLinkBuilder:
    def __init__(
        self,
        layout: SampleLayout,
        distortion_payload: dict,
        profile_name: str,
    ):
        self.layout = layout
        self.distortion_payload = distortion_payload
        self.profile_name = profile_name

    def build(
        self,
        recipe: str,
        total_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        keep_top: int | None = None,
    ) -> torch.Tensor:
        builders = {
            "ref_ref": self._reference_mesh,
            "ref_dist": self._reference_to_distortion,
            "dist_dist": self._within_distortion_family,
            "structural": self._paired_view_alignment,
        }
        try:
            builder = builders[recipe]
        except KeyError as exc:
            raise ValueError(f"Unsupported metadata relation '{recipe}'") from exc
        return builder(total_nodes, device, dtype, keep_top)

    def _empty(self, total_nodes: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty(0, dtype=dtype, device=device),
            (total_nodes, total_nodes),
        ).coalesce()

    def _pack(self, row_ids, col_ids, weights, total_nodes: int) -> torch.Tensor:
        return torch.sparse_coo_tensor(
            torch.stack((row_ids, col_ids)),
            weights,
            (total_nodes, total_nodes),
        ).coalesce()

    def _reference_mesh(
        self,
        total_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        _keep_top: int | None,
    ) -> torch.Tensor:
        ref_ids = torch.arange(self.layout.reference_nodes, device=device)
        pairs = torch.cartesian_prod(ref_ids, ref_ids)
        keep = pairs[:, 0] != pairs[:, 1]
        pairs = pairs[keep]
        weights = torch.ones(int(pairs.shape[0]), dtype=dtype, device=device)
        return self._pack(pairs[:, 0], pairs[:, 1], weights, total_nodes)

    def _reference_to_distortion(
        self,
        total_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        _keep_top: int | None,
    ) -> torch.Tensor:
        batch_count, view_count, family_count, level_count = self.layout.distortion_axes
        index_grid = torch.cartesian_prod(
            torch.arange(batch_count, device=device),
            torch.arange(view_count, device=device),
            torch.arange(family_count, device=device),
            torch.arange(level_count, device=device),
        )
        sample_ids, view_ids, family_ids, level_ids = index_grid.unbind(dim=1)

        src_nodes = self.layout.ref_node(sample_ids, view_ids)
        dst_nodes = self.layout.dist_node(sample_ids, view_ids, family_ids, level_ids)

        active_axis = self.distortion_payload["var_dist"][sample_ids, family_ids]
        level_table = self.distortion_payload["indices"][sample_ids, family_ids, level_ids]
        active_levels = level_table.gather(1, active_axis.unsqueeze(1)).squeeze(1)
        weights = severity_to_strength(active_levels, self.profile_name).to(dtype=dtype)
        return self._pack(src_nodes, dst_nodes, weights, total_nodes)

    def _within_distortion_family(
        self,
        total_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        keep_top: int | None,
    ) -> torch.Tensor:
        batch_count, view_count, family_count, level_count = self.layout.distortion_axes
        index_grid = torch.cartesian_prod(
            torch.arange(batch_count, device=device),
            torch.arange(view_count, device=device),
            torch.arange(family_count, device=device),
            torch.arange(level_count, device=device),
            torch.arange(view_count, device=device),
            torch.arange(level_count, device=device),
        )
        sample_ids, left_views, family_ids, left_levels, right_views, right_levels = index_grid.unbind(dim=1)

        left_nodes = self.layout.dist_node(sample_ids, left_views, family_ids, left_levels)
        right_nodes = self.layout.dist_node(sample_ids, right_views, family_ids, right_levels)
        keep = left_nodes != right_nodes
        if not bool(keep.any()):
            return self._empty(total_nodes, device, dtype)

        sample_ids = sample_ids[keep]
        family_ids = family_ids[keep]
        left_levels = left_levels[keep]
        right_levels = right_levels[keep]
        left_nodes = left_nodes[keep]
        right_nodes = right_nodes[keep]

        active_axis = self.distortion_payload["var_dist"][sample_ids, family_ids]
        left_table = self.distortion_payload["indices"][sample_ids, family_ids, left_levels]
        right_table = self.distortion_payload["indices"][sample_ids, family_ids, right_levels]
        left_active = left_table.gather(1, active_axis.unsqueeze(1)).squeeze(1)
        right_active = right_table.gather(1, active_axis.unsqueeze(1)).squeeze(1)
        weights = severity_to_strength(
            (left_active - right_active).abs(),
            self.profile_name,
        ).to(dtype=dtype)

        if keep_top is not None and int(keep_top) < int(weights.numel()):
            weights, positions = torch.topk(weights, int(keep_top), sorted=False)
            left_nodes = left_nodes[positions]
            right_nodes = right_nodes[positions]

        return self._pack(left_nodes, right_nodes, weights, total_nodes)

    def _paired_view_alignment(
        self,
        total_nodes: int,
        device: torch.device,
        dtype: torch.dtype,
        _keep_top: int | None,
    ) -> torch.Tensor:
        batch_count, view_count, family_count, level_count = self.layout.distortion_axes
        index_grid = torch.cartesian_prod(
            torch.arange(batch_count, device=device),
            torch.arange(view_count, device=device),
            torch.arange(family_count, device=device),
            torch.arange(level_count, device=device),
            torch.arange(view_count, device=device),
        )
        sample_ids, left_views, family_ids, level_ids, right_views = index_grid.unbind(dim=1)

        keep = (left_views != right_views) & ((left_views % 2) == (right_views % 2))
        if not bool(keep.any()):
            return self._empty(total_nodes, device, dtype)

        left_nodes = self.layout.dist_node(
            sample_ids[keep],
            left_views[keep],
            family_ids[keep],
            level_ids[keep],
        )
        right_nodes = self.layout.dist_node(
            sample_ids[keep],
            right_views[keep],
            family_ids[keep],
            level_ids[keep],
        )
        weights = torch.ones(int(left_nodes.numel()), dtype=dtype, device=device)
        return self._pack(left_nodes, right_nodes, weights, total_nodes)
