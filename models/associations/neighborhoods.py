import torch
import torch.nn.functional as F


def _dense_affinity(embeddings: torch.Tensor, metric_name: str) -> torch.Tensor:
    if metric_name == "euc":
        distances = torch.cdist(embeddings, embeddings)
        affinity = torch.reciprocal(1.0 + distances)
    elif metric_name == "cos":
        normalized = F.normalize(embeddings, dim=-1)
        affinity = normalized @ normalized.T
    else:
        raise ValueError(f"Unsupported neighbor metric '{metric_name}'")

    eye = torch.eye(affinity.size(0), dtype=torch.bool, device=affinity.device)
    return affinity.masked_fill(eye, 0.0)


def build_neighbor_links(
    embeddings: torch.Tensor,
    neighbor_count: int,
    metric_name: str,
    include_self: bool,
    symmetric: bool,
    soft_entries: bool,
) -> torch.Tensor:
    total_nodes = int(embeddings.size(0))
    affinity = _dense_affinity(embeddings, metric_name)
    if include_self:
        eye = torch.eye(total_nodes, dtype=torch.bool, device=embeddings.device)
        affinity = affinity.masked_fill(eye, 1.0)

    neighbor_count = min(int(neighbor_count), total_nodes)
    if neighbor_count <= 0:
        return torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long, device=embeddings.device),
            torch.empty(0, dtype=embeddings.dtype, device=embeddings.device),
            (total_nodes, total_nodes),
        ).coalesce()

    link_values, link_cols = torch.topk(
        affinity,
        neighbor_count,
        dim=1,
        largest=True,
        sorted=False,
    )
    link_rows = torch.arange(total_nodes, device=embeddings.device).repeat_interleave(
        neighbor_count
    )
    link_cols = link_cols.reshape(-1)
    link_values = link_values.reshape(-1)

    if not soft_entries:
        link_values = torch.ones_like(link_values)

    if symmetric:
        reverse_rows = link_cols
        reverse_cols = link_rows
        link_rows = torch.cat((link_rows, reverse_rows))
        link_cols = torch.cat((link_cols, reverse_cols))
        link_values = link_values.repeat(2)

    return torch.sparse_coo_tensor(
        torch.stack((link_rows, link_cols)),
        link_values,
        (total_nodes, total_nodes),
    ).coalesce()
