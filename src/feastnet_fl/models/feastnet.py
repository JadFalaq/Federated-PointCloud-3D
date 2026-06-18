import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import BatchNorm, FeaStConv, global_max_pool, global_mean_pool


def build_knn_edge_index(pos: torch.Tensor, batch: torch.Tensor, k: int) -> torch.Tensor:
    edge_indices = []
    device = pos.device

    for graph_id in batch.unique(sorted=True):
        node_indices = torch.where(batch == graph_id)[0]
        graph_pos = pos[node_indices]
        num_nodes = graph_pos.size(0)
        if num_nodes <= 1:
            continue

        effective_k = min(k, num_nodes - 1)
        distances = torch.cdist(graph_pos, graph_pos, p=2)
        distances.fill_diagonal_(float("inf"))
        neighbors = distances.topk(k=effective_k, largest=False).indices

        source = node_indices.repeat_interleave(effective_k)
        target = node_indices[neighbors.reshape(-1)]
        edge_indices.append(torch.stack([source, target], dim=0))

    if not edge_indices:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.cat(edge_indices, dim=1)


class FeaStNetClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels=(64, 128, 256),
        heads=(4, 4, 8),
        k: int = 16,
        dropout: float = 0.3,
        num_classes: int = 50,
    ) -> None:
        super().__init__()
        if len(hidden_channels) != len(heads):
            raise ValueError("hidden_channels and heads must have the same length")

        self.k = k
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        last_channels = in_channels
        for out_channels, head_count in zip(hidden_channels, heads):
            self.convs.append(FeaStConv(last_channels, out_channels, heads=head_count))
            self.norms.append(BatchNorm(out_channels))
            last_channels = out_channels

        pooled_channels = hidden_channels[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(pooled_channels, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, data):
        x = data.x if getattr(data, "x", None) is not None else data.pos
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.pos.size(0), dtype=torch.long, device=data.pos.device)
        edge_index = build_knn_edge_index(data.pos, batch=batch, k=self.k)

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return self.classifier(pooled)


class StudentFeaStNet(FeaStNetClassifier):
    def __init__(
        self,
        in_channels: int = 6,
        hidden_channels=(32, 64, 64),
        heads=(2, 2, 4),
        k: int = 12,
        dropout: float = 0.2,
        num_classes: int = 50,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            heads=heads,
            k=k,
            dropout=dropout,
            num_classes=num_classes,
        )
