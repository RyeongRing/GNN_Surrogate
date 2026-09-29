"""
Physics-informed GNN Surrogate 본체.
encoder-processor-decoder 구조 + continuity loss.
"""

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.error("torch / torch_geometric not installed.")
    # 하위 클래스 정의를 위한 placeholder
    class nn:
        class Module:
            pass


class RainfallEncoder(nn.Module):
    """강우 hyetograph [B, T_rain, 1] → summary [B, d_rain] + temporal [B, T_out, d_rain]."""

    def __init__(self, d_rain: int = 32, n_layers: int = 1,
                 T_rain: int = 72, T_out: int = 100):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=d_rain,
                            num_layers=n_layers, batch_first=True)
        # Project T_rain → T_out time steps (align rain and output axes)
        self.time_align = nn.Linear(T_rain, T_out, bias=False)

    def forward(self, rain: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        lstm_out, (h_n, _) = self.lstm(rain)           # lstm_out: [B, T_rain, d_rain]
        summary = h_n[-1]                               # [B, d_rain]
        # Temporal features: project T_rain → T_out
        temporal = self.time_align(lstm_out.transpose(1, 2))  # [B, d_rain, T_out]
        temporal = temporal.transpose(1, 2)             # [B, T_out, d_rain]
        return summary, temporal


class NodeEncoder(nn.Module):
    """노드 정적 피처 + 강우 latent → 초기 노드 임베딩."""

    def __init__(self, node_feat_dim: int, d_rain: int, hidden_dim: int,
                 dropout: float = 0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(node_feat_dim + d_rain, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_node: "torch.Tensor",
                rain_emb: "torch.Tensor") -> "torch.Tensor":
        n_nodes = x_node.size(0)
        rain_expand = rain_emb.unsqueeze(0).expand(n_nodes, -1)
        return self.fc(torch.cat([x_node, rain_expand], dim=-1))


class GraphProcessor(nn.Module):
    """L개의 GATv2Conv (또는 GCNConv) 레이어 + residual connection.

    conv_type="gat" (default): GATv2Conv, edge_attr[N,4] 사용 (attention weighting).
    conv_type="gcn": plain GCNConv — topology + multi-hop message passing은 유지하되
        attention이 없는 ablation arm (attention 유무를 isolate하기 위한 baseline).
    """

    def __init__(self, hidden_dim: int, n_heads: int, n_layers: int,
                 edge_feat_dim: int = 4, dropout: float = 0.0,
                 conv_type: str = "gat"):
        super().__init__()
        self.conv_type = conv_type
        if conv_type == "gat":
            self.layers = nn.ModuleList([
                GATv2Conv(hidden_dim, hidden_dim // n_heads,
                          heads=n_heads, edge_dim=edge_feat_dim,
                          concat=True, add_self_loops=False)
                for _ in range(n_layers)
            ])
        elif conv_type == "gcn":
            # NOTE for manuscript: GCNConv only accepts an optional *scalar*
            # edge_weight, unlike GATv2Conv's multi-dimensional edge_dim=4
            # (pipe length/diameter/slope/roughness). We drop edge_attr
            # entirely here (the standard "vanilla GCN" baseline convention
            # in the GNN literature) rather than reduce it to a scalar, so
            # the MESSAGE-PASSING layers in this arm have NO access to those
            # 4 edge features during multi-hop propagation. IMPORTANT: this
            # does NOT mean the GCN arm is edge-feature-blind overall —
            # HydraulicDecoder.link_head (see forward() below and
            # HydraulicDecoder.forward) concatenates edge_attr directly and
            # receives it identically regardless of conv_type. Also note
            # GCNConv is NOT parameter-matched to GATv2Conv at the same
            # hidden_dim/n_layers (roughly 165k vs 234k total model
            # parameters at hidden_dim=128/n_layers=4/n_heads=4) and uses
            # PyG's default add_self_loops=True + symmetric degree
            # normalisation, whereas the GAT branch below explicitly sets
            # add_self_loops=False. This is therefore a practical
            # operator-substitution comparison (GCNConv vs. GATv2Conv as a
            # drop-in replacement at matched depth/width), not a controlled,
            # single-factor attention-only ablation — state this explicitly
            # wherever this comparison is reported in the manuscript.
            self.layers = nn.ModuleList([
                GCNConv(hidden_dim, hidden_dim)
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(f"Unknown conv_type: {conv_type!r} (expected 'gat' or 'gcn')")
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor",
                edge_attr: "torch.Tensor") -> "torch.Tensor":
        for conv, norm in zip(self.layers, self.norms):
            if self.conv_type == "gcn":
                h = conv(x, edge_index)  # edge_attr intentionally dropped (see NOTE above)
            else:
                h = conv(x, edge_index, edge_attr)
            x = norm(x + h)
            x = self.dropout(x)
        return x


class HydraulicDecoder(nn.Module):
    """노드 임베딩 → 수위 시계열 [N, T] + 링크 유량 [E, T]. (MLP, Loop ≤6용 legacy)"""

    def __init__(self, hidden_dim: int, T_out: int, edge_feat_dim: int = 4):
        super().__init__()
        self.T_out = T_out
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )

    def forward(self, x_node: "torch.Tensor",
                edge_index: "torch.Tensor",
                edge_attr: "torch.Tensor",
                y_true: Optional["torch.Tensor"] = None,
                ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        depth = self.node_head(x_node)  # [N, T]
        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([x_node[src], x_node[dst], edge_attr], dim=-1)
        flow = self.link_head(edge_input)  # [E, T]
        return depth, flow


class RainConditionedDecoder(nn.Module):
    """
    강우-조건부 시간 디코더 (Loop 13).

    depth = MLP(x_node) + time_proj(x_node) @ rain_temporal.T
            [N, T]      + [N, d_rain]        @ [d_rain, T]  = [N, T]

    rain_temporal: RainfallEncoder에서 넘어온 실제 강우 시계열 특징 [T_out, d_rain].
    time_proj(x_node)는 노드별 강우 특징 가중치 → 각 노드가 언제 응답할지 학습.
    """

    def __init__(self, hidden_dim: int, T_out: int, edge_feat_dim: int = 4,
                 d_rain: int = 32):
        super().__init__()
        self.T_out = T_out
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )
        # Per-node projection into rain feature space — zeros init so model starts
        # as pure MLP decoder; gradient from rain_temporal (nonzero) pushes it.
        self.time_proj = nn.Linear(hidden_dim, d_rain, bias=False)
        nn.init.zeros_(self.time_proj.weight)
        # Link flow head
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )

    def forward(self, x_node: "torch.Tensor",
                edge_index: "torch.Tensor",
                edge_attr: "torch.Tensor",
                rain_temporal: Optional["torch.Tensor"] = None,
                y_true: Optional["torch.Tensor"] = None,
                ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        depth_base = self.node_head(x_node)              # [N, T]
        if rain_temporal is not None:
            rt = rain_temporal.squeeze(0)                 # [T_out, d_rain]
            temporal_w = self.time_proj(x_node)          # [N, d_rain]
            depth_rain = temporal_w @ rt.T               # [N, T_out]
            depth = depth_base + depth_rain
        else:
            depth = depth_base

        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([x_node[src], x_node[dst], edge_attr], dim=-1)
        flow = self.link_head(edge_input)                # [E, T]
        return depth, flow


class ScalarRainDecoder(nn.Module):
    """
    Per-node scalar rain-sensitivity decoder (Loop 20).

    depth = MLP(x_node) + sensitivity(x_node) * rain_profile(t)
           [N, T]      +  [N, 1]              * [T]           = [N, T]

    rain_profile(t) = mean_d(rain_temporal[t, :]) — temporal rain signal, shape [T].
    sensitivity: Linear(hidden_dim, 1, bias=False), zeros-init — only 128 extra params.
    Model starts as pure MLP; gradient from rain_profile gradually activates sensitivity.
    """

    def __init__(self, hidden_dim: int, T_out: int, edge_feat_dim: int = 4):
        super().__init__()
        self.T_out = T_out
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )
        self.sensitivity = nn.Linear(hidden_dim, 1, bias=False)
        nn.init.zeros_(self.sensitivity.weight)  # start as pure MLP, let grad activate
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )

    def forward(self, x_node: "torch.Tensor",
                edge_index: "torch.Tensor",
                edge_attr: "torch.Tensor",
                rain_temporal: Optional["torch.Tensor"] = None,
                y_true: Optional["torch.Tensor"] = None,
                ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        depth = self.node_head(x_node)                    # [N, T]
        if rain_temporal is not None:
            rt = rain_temporal.squeeze(0)                 # [T_out, d_rain]
            rain_profile = rt.mean(dim=-1)               # [T_out]
            sens = self.sensitivity(x_node)               # [N, 1]
            depth = depth + sens * rain_profile.unsqueeze(0)  # [N, T]
        src, dst = edge_index[0], edge_index[1]
        flow = self.link_head(
            torch.cat([x_node[src], x_node[dst], edge_attr], dim=-1))  # [E, T]
        return depth, flow


class TemporalBilinearDecoder(nn.Module):
    """
    저-랭크 이중선형 시간 디코더 (Loop 8).

    depth = MLP(x_node) + time_proj(x_node) @ time_emb.T
            [N, T]      + [N, d_time]        @ [d_time, T]  = [N, T]

    핵심: time_proj(x_node) 는 각 노드의 '시간 지문'(temporal fingerprint),
    time_emb 는 T개 학습된 시간 기저 함수.
    저장형 노드 vs 통과형 노드가 서로 다른 시간 반응 패턴을 가짐을 모델링.
    LSTM 없이 완전 병렬 — N=8480, T=100 에서 MLP 대비 추가 메모리 < 5MB.
    """

    def __init__(self, hidden_dim: int, T_out: int, edge_feat_dim: int = 4,
                 d_time: int = 32):
        super().__init__()
        self.T_out = T_out
        # Base MLP (동일 구조, MLP decoder와 같음)
        self.node_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )
        # 저-랭크 시간 보정항
        self.time_proj = nn.Linear(hidden_dim, d_time, bias=False)
        self.time_emb  = nn.Parameter(torch.zeros(T_out, d_time))
        nn.init.normal_(self.time_emb, std=0.01)
        # 링크 유량 MLP
        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, T_out),
        )

    def forward(self, x_node: "torch.Tensor",
                edge_index: "torch.Tensor",
                edge_attr: "torch.Tensor",
                y_true: Optional["torch.Tensor"] = None,
                ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        depth_base = self.node_head(x_node)              # [N, T]
        time_logits = self.time_proj(x_node)             # [N, d_time]
        time_corr   = time_logits @ self.time_emb.T      # [N, T]
        depth = depth_base + time_corr                   # [N, T]

        src, dst = edge_index[0], edge_index[1]
        edge_input = torch.cat([x_node[src], x_node[dst], edge_attr], dim=-1)
        flow = self.link_head(edge_input)                # [E, T]
        return depth, flow


class PhysicsInformedGNN(nn.Module):
    """
    Physics-informed GNN Surrogate 메인 클래스.
    use_physics_loss=False 플래그로 'graph w/o physics' baseline 전환.
    """

    def __init__(self, node_feat_dim: int, edge_feat_dim: int,
                 hidden_dim: int = 128, T_out: int = 100, T_rain: int = 72,
                 n_heads: int = 4, n_layers: int = 4,
                 d_rain: int = 32, use_physics_loss: bool = True,
                 dropout: float = 0.0,
                 temporal_decoder: bool = True,
                 scalar_rain_decoder: bool = False,
                 conv_type: str = "gat"):
        super().__init__()
        self.use_physics_loss = use_physics_loss
        self.rain_encoder = RainfallEncoder(d_rain=d_rain, T_rain=T_rain, T_out=T_out)
        self.node_encoder = NodeEncoder(node_feat_dim, d_rain, hidden_dim, dropout=dropout)
        self.processor = GraphProcessor(hidden_dim, n_heads, n_layers, edge_feat_dim,
                                        dropout=dropout, conv_type=conv_type)
        if scalar_rain_decoder:
            self.decoder = ScalarRainDecoder(hidden_dim, T_out, edge_feat_dim)
        elif temporal_decoder:
            self.decoder = RainConditionedDecoder(hidden_dim, T_out, edge_feat_dim, d_rain)
        else:
            self.decoder = HydraulicDecoder(hidden_dim, T_out, edge_feat_dim)

    def forward(self, data: "Data",
                teacher_forcing: bool = False) -> Dict[str, "torch.Tensor"]:
        rain_summary, rain_temporal = self.rain_encoder(data.rain.unsqueeze(0))
        # rain_summary: [1, d_rain], rain_temporal: [1, T_out, d_rain]
        x = self.node_encoder(data.x, rain_summary.squeeze(0))
        x = self.processor(x, data.edge_index, data.edge_attr)
        y_true = data.y if teacher_forcing else None
        if isinstance(self.decoder, (RainConditionedDecoder, ScalarRainDecoder)):
            depth_pred, flow_pred = self.decoder(
                x, data.edge_index, data.edge_attr,
                rain_temporal=rain_temporal, y_true=y_true)
        else:
            depth_pred, flow_pred = self.decoder(x, data.edge_index, data.edge_attr,
                                                  y_true=y_true)
        return {"depth_pred": depth_pred, "flow_pred": flow_pred, "node_emb": x}

    def compute_continuity_loss(self, depth_pred: "torch.Tensor",
                                 flow_pred: "torch.Tensor",
                                 edge_index: "torch.Tensor",
                                 dt: Optional["torch.Tensor"] = None,
                                 storage_area: Optional["torch.Tensor"] = None,
                                 S_ref: float = 1.0) -> "torch.Tensor":
        """
        연속방정식 소프트 패널티:
          r_i(t) = |A_i * (h_i(t+1) - h_i(t)) / dt - (Q_in_i(t) - Q_out_i(t))| / S_ref
          L_cont = mean(r_i)

        depth_pred: [N, T]
        flow_pred:  [E, T]
        """
        N, T = depth_pred.shape
        if dt is None:
            raise ValueError("dt_seconds is required for continuity loss")
        dt_values = torch.as_tensor(
            dt, dtype=depth_pred.dtype, device=depth_pred.device).reshape(-1)
        if dt_values.numel() != 1:
            raise ValueError(
                "continuity loss currently requires batch_size=1 so each sample has one dt_seconds value"
            )
        dt_seconds = dt_values[0]
        if not torch.isfinite(dt_seconds).item() or dt_seconds.item() <= 0:
            raise ValueError(f"dt_seconds must be finite and positive, got {dt_seconds.item()}")
        if storage_area is None:
            storage_area = torch.ones(N, 1, device=depth_pred.device)

        dS_dt = storage_area * (depth_pred[:, 1:] - depth_pred[:, :-1]) / dt_seconds  # [N, T-1]

        src, dst = edge_index[0], edge_index[1]
        Q_in = torch.zeros(N, T - 1, device=flow_pred.device)
        Q_out = torch.zeros(N, T - 1, device=flow_pred.device)
        Q_in.index_add_(0, dst, flow_pred[:, :T - 1])
        Q_out.index_add_(0, src, flow_pred[:, :T - 1])

        residual = torch.abs(dS_dt - (Q_in - Q_out)) / S_ref
        return residual.mean()

    def compute_total_loss(self, pred: Dict, target: "torch.Tensor",
                           edge_index: "torch.Tensor",
                           dt_seconds: Optional["torch.Tensor"] = None,
                           storage_area: Optional["torch.Tensor"] = None,
                           lambda_cont: float = 0.1,
                           max_depth: Optional["torch.Tensor"] = None,
                           alpha_peak: float = 0.0,
                           use_nse_loss: bool = False) -> "torch.Tensor":
        """L_total = L_depth + lambda_cont * L_cont
        L_depth options: uniform MSE | peak-weighted MSE | direct 1-NSE
        """
        if max_depth is not None:
            md = max_depth.unsqueeze(-1)
            L_depth = F.mse_loss(pred["depth_pred"] / md, target / md)
        elif use_nse_loss:
            # Direct NSE loss: L = Σ(y-ŷ)² / max(Σ(y-ȳ)², σ²_min)
            # Directly minimizes 1-NSE per event → maximizes NSE at test time.
            err_sq = (pred["depth_pred"] - target) ** 2
            y_bar = target.mean(dim=-1, keepdim=True)        # [N, 1] per-node mean
            var_y = ((target - y_bar) ** 2).sum()             # scalar per event
            L_depth = err_sq.sum() / (var_y + 1e-4)
        elif alpha_peak > 0.0:
            peak_w = 1.0 + alpha_peak * (target / (target.amax(dim=-1, keepdim=True) + 1e-6))
            L_depth = (peak_w * (pred["depth_pred"] - target) ** 2).mean()
        else:
            L_depth = F.mse_loss(pred["depth_pred"], target)
        if self.use_physics_loss:
            L_cont = self.compute_continuity_loss(
                pred["depth_pred"], pred["flow_pred"],
                edge_index, dt=dt_seconds, storage_area=storage_area
            )
            return L_depth + lambda_cont * L_cont
        return L_depth


class EnsembleGNN(nn.Module):
    """Deep Ensemble 래퍼 (M개 독립 PhysicsInformedGNN)."""

    def __init__(self, base_config: Dict, M: int = 5):
        super().__init__()
        self.M = M
        self.members = nn.ModuleList([
            PhysicsInformedGNN(**base_config) for _ in range(M)
        ])

    def forward(self, data: "Data") -> Dict[str, "torch.Tensor"]:
        """Average member predictions; the final MLP decoder emits all T steps at once."""
        depth_preds = []
        flow_preds = []
        for member in self.members:
            out = member(data, teacher_forcing=False)
            depth_preds.append(out["depth_pred"])
            flow_preds.append(out["flow_pred"])

        depth_stack = torch.stack(depth_preds, dim=0)  # [M, N, T]
        flow_stack = torch.stack(flow_preds, dim=0)    # [M, E, T]
        return {
            "mean_depth": depth_stack.mean(0),
            "std_depth":  depth_stack.std(0),
            "mean_flow":  flow_stack.mean(0),
            "std_flow":   flow_stack.std(0),
            "depth_stack": depth_stack,
            "flow_stack":  flow_stack,
        }

    def compute_total_loss(self, data: "Data", lambda_cont: float = 0.1,
                           alpha_peak: float = 0.0,
                           use_nse_loss: bool = False) -> "torch.Tensor":
        """Average the members\' depth and auxiliary internal-consistency losses."""
        dt_seconds = getattr(data, "dt_seconds", None)
        if dt_seconds is None:
            raise AttributeError("SWMMGraphDataset sample is missing dt_seconds")
        losses = []
        for member in self.members:
            pred = member(data, teacher_forcing=True)
            loss = member.compute_total_loss(
                pred, data.y, data.edge_index, dt_seconds=dt_seconds,
                lambda_cont=lambda_cont, max_depth=None, alpha_peak=alpha_peak,
                use_nse_loss=use_nse_loss,
            )
            losses.append(loss)
        return torch.stack(losses).mean()
