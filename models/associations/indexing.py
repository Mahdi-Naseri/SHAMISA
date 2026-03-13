class SampleLayout:
    def __init__(
        self,
        z_ref,
        z_dist,
        _h_ref,
        _h_dist,
        z_ref_compact,
        *_unused,
    ):
        self._ref_view_count = int(z_ref.shape[1])
        self._ref_node_count = int(z_ref_compact.shape[0])
        batch_count, view_count, family_count, level_count = z_dist.shape[:4]
        self._distortion_axes = (
            int(batch_count),
            int(view_count),
            int(family_count),
            int(level_count),
        )
        self._family_stride = self._distortion_axes[3]
        self._view_stride = self._distortion_axes[2] * self._family_stride
        self._batch_stride = self._distortion_axes[1] * self._view_stride

    @property
    def reference_nodes(self) -> int:
        return self._ref_node_count

    @property
    def distortion_axes(self) -> tuple[int, int, int, int]:
        return self._distortion_axes

    def ref_offset(self, sample_id, view_id):
        return sample_id * self._ref_view_count + view_id

    def dist_offset(self, sample_id, view_id, family_id, level_id):
        return (
            sample_id * self._batch_stride
            + view_id * self._view_stride
            + family_id * self._family_stride
            + level_id
        )

    def ref_node(self, sample_id, view_id):
        return self.ref_offset(sample_id, view_id)

    def dist_node(self, sample_id, view_id, family_id, level_id):
        return self.reference_nodes + self.dist_offset(
            sample_id,
            view_id,
            family_id,
            level_id,
        )
