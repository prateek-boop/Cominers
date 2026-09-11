"""Phase 2: Dynamic Temporal Graph Construction.

Represents network communication as a continuous-time directed multigraph
G_t = (V_t, E_t) partitioned into discrete snapshot intervals Delta t = 15 seconds.
Enforces static role-based node merging to prevent dynamic port explosion:
- Client (ephemeral dynamic ports aggregated)
- Server / Sensitive Service (e.g., 445-SMB, 22-SSH, 80/443-Web)
- Router / Core Admin Infrastructure
- Lawful Intercept / CALEA Wiretap Core
"""
from dataclasses import dataclass, field
from copy import deepcopy
import time
from typing import Dict, List, Set, Tuple
import numpy as np
import torch

from features.dual_branch import DualBranchFeatureExtractor


ROLE_CLIENT = 0
ROLE_SERVER = 1
ROLE_ROUTER_ADMIN = 2
ROLE_SENSITIVE_CALEA = 3


@dataclass
class DynamicNode:
    node_id: int
    ip: str
    role: int
    role_name: str
    in_degree: int = 0
    out_degree: int = 0
    total_packets: int = 0
    internal_connections: int = 0
    external_connections: int = 0

    def compute_feature_vector(self) -> np.ndarray:
        tot_conn = max(1, self.internal_connections + self.external_connections)
        ratio = float(self.internal_connections / tot_conn)
        return DualBranchFeatureExtractor.extract_node_feature_vector_16(
            in_degree=self.in_degree,
            out_degree=self.out_degree,
            internal_external_ratio=ratio,
            packet_volume_delta=float(self.total_packets),
            role_id=self.role,
        )


@dataclass
class DynamicEdge:
    src_node_id: int
    dst_node_id: int
    timestamp: float
    feature_vector_32: np.ndarray


@dataclass
class TemporalGraphSnapshot:
    window_index: int
    start_time: float
    end_time: float
    nodes: Dict[int, DynamicNode]
    edges: List[DynamicEdge]

    def to_pyg_data(self) -> dict:
        """Convert snapshot into PyTorch tensors."""
        num_nodes = len(self.nodes)
        node_features = np.zeros((num_nodes, 16), dtype=np.float32)
        node_id_map = {}

        for idx, (node_key, node) in enumerate(self.nodes.items()):
            node_id_map[node.node_id] = idx
            node_features[idx] = node.compute_feature_vector()

        if self.edges:
            src_list = [node_id_map.get(e.src_node_id, 0) for e in self.edges]
            dst_list = [node_id_map.get(e.dst_node_id, 0) for e in self.edges]
            times = [e.timestamp for e in self.edges]
            edge_feats = [e.feature_vector_32 for e in self.edges]

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr = torch.tensor(np.array(edge_feats), dtype=torch.float32)
            edge_times = torch.tensor(times, dtype=torch.float32)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 32), dtype=torch.float32)
            edge_times = torch.empty(0, dtype=torch.float32)

        return {
            "x": torch.tensor(node_features, dtype=torch.float32),
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "edge_time": edge_times,
            "num_nodes": num_nodes,
        }


class TemporalGraphBuilder:
    """Constructs 15-second discrete temporal graph snapshots from streaming interactions."""

    def __init__(self, snapshot_interval_sec: float = 15.0):
        self.interval_sec = snapshot_interval_sec
        self.ip_to_node: Dict[str, DynamicNode] = {}
        self.active_edges: List[DynamicEdge] = []
        self.window_index = 0
        self.current_window_start = time.time()

    def _determine_role(self, ip: str, port: int) -> Tuple[int, str]:
        """Determine static role without ephemeral port explosion."""
        # Detect telecom / CALEA wiretap or sensitive internal management targets
        if "calea" in ip.lower() or ip.endswith(".250") or port == 8443:
            return ROLE_SENSITIVE_CALEA, "CALEA-Wiretap-Core"
        if port in (22, 23, 80, 443, 8080) and (ip.startswith("10.0.") or ip.endswith(".1")):
            return ROLE_ROUTER_ADMIN, "Router-Admin-Core"
        if port in (445, 139, 3389, 3306, 5432):
            return ROLE_SERVER, f"Server-{port}"
        if port >= 1024:
            return ROLE_CLIENT, "Client"
        return ROLE_SERVER, f"Service-{port}"

    def get_or_register_node(self, ip: str, port: int) -> DynamicNode:
        if ip not in self.ip_to_node:
            role_id, role_name = self._determine_role(ip, port)
            node_id = len(self.ip_to_node)
            self.ip_to_node[ip] = DynamicNode(
                node_id=node_id,
                ip=ip,
                role=role_id,
                role_name=role_name,
            )
        return self.ip_to_node[ip]

    def add_interaction(self, src_ip: str, dst_ip: str, src_port: int, dst_port: int, timestamp: float, feature_32: np.ndarray):
        src_node = self.get_or_register_node(src_ip, src_port)
        dst_node = self.get_or_register_node(dst_ip, dst_port)

        src_node.out_degree += 1
        src_node.total_packets += 1
        dst_node.in_degree += 1
        dst_node.total_packets += 1

        is_internal = dst_ip.startswith("10.") or dst_ip.startswith("192.168.")
        if is_internal:
            src_node.internal_connections += 1
        else:
            src_node.external_connections += 1

        edge = DynamicEdge(
            src_node_id=src_node.node_id,
            dst_node_id=dst_node.node_id,
            timestamp=timestamp,
            feature_vector_32=np.array(feature_32, copy=True),
        )
        self.active_edges.append(edge)

    def close_snapshot(self) -> TemporalGraphSnapshot:
        """Freeze current 15-second snapshot and advance window."""
        snapshot = TemporalGraphSnapshot(
            window_index=self.window_index,
            start_time=self.current_window_start,
            end_time=self.current_window_start + self.interval_sec,
            nodes={n.node_id: deepcopy(n) for n in self.ip_to_node.values()},
            edges=list(self.active_edges),
        )
        self.window_index += 1
        self.current_window_start += self.interval_sec
        self.active_edges.clear()
        return snapshot
