"""Streaming CyberTGN Runtime with persistent node memory and sliding-window history.

Maintains host state across requests/batches, solving the cold-start amnesia
where the original inference reset state on every single HTTP call.
"""
from __future__ import annotations
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import torch
from torch_geometric.nn.models.tgn import LastNeighborLoader
from models.current_flow import load_current_flow, combined_logits, encode_timestamps
from models.memory import MemoryModule
from models.tgn import CyberTGN
from models.multitask_heads import AttackProbabilityHead, MitreStageClassifier

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / 'checkpoints/tgn_best.pt'
FEATURE_NAMES = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packets/s', 'Bwd Packets/s', 'Average Packet Size', 'Down/Up Ratio',
]
STAGES = ['benign', 'reconnaissance', 'initial_access', 'credential_access',
          'lateral_movement', 'command_and_control', 'exfiltration', 'impact']


class PersistentIPMapper:
    """Maintains an LRU IP-to-NodeID table capped at maximum graph node capacity."""

    def __init__(self, max_nodes: int):
        if max_nodes < 1:
            raise ValueError("max_nodes must be positive")
        self.max_nodes = max_nodes
        self.ip_to_id: OrderedDict[str, int] = OrderedDict()
        self.id_to_ip: Dict[int, str] = {}
        self.free_ids: List[int] = list(reversed(range(max_nodes)))

    def get_or_create(self, ip: str) -> Tuple[int, bool]:
        """Returns (node_id, is_new)."""
        if ip in self.ip_to_id:
            self.ip_to_id.move_to_end(ip)
            return self.ip_to_id[ip], False

        if self.free_ids:
            node_id = self.free_ids.pop()
        else:
            # Evict least recently active IP
            old_ip, node_id = self.ip_to_id.popitem(last=False)
            del self.id_to_ip[old_ip]

        self.ip_to_id[ip] = node_id
        self.id_to_ip[node_id] = ip
        return node_id, True

    def contains(self, ip: str) -> bool:
        return ip in self.ip_to_id

    def clear(self):
        self.ip_to_id.clear()
        self.id_to_ip.clear()
        self.free_ids = list(reversed(range(self.max_nodes)))


class StreamingModelRuntime:
    """Thread-safe persistent CyberTGN inference runtime for real-time traffic streams."""

    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, max_history=100000, threads=2):
        if not isinstance(max_history, int) or max_history < 1:
            raise ValueError("max_history must be positive")
        torch.set_num_threads(threads)
        self.path = Path(checkpoint).resolve()
        ckpt = torch.load(self.path, map_location='cpu', weights_only=True)

        mem = ckpt['memory']
        self.num_nodes, dim = mem['memory.memory'].shape
        time_dim = mem['memory.time_enc.lin.weight'].shape[0]
        self.msg_dim = mem['memory.gru.weight_ih'].shape[1] - 2 * dim - time_dim
        embedding_dim = ckpt['prob_head']['mlp.0.weight'].shape[1] // 2
        self.num_classes = ckpt['mitre_head']['mlp.2.weight'].shape[0]

        time_transform = ckpt.get('model_config',{}).get('time_transform','identity')
        self.memory = MemoryModule(self.num_nodes, self.msg_dim, dim, time_dim,time_transform)
        self.gnn = CyberTGN(self.memory, dim, embedding_dim, self.msg_dim, time_dim,time_transform)
        self.prob_head = AttackProbabilityHead(embedding_dim)
        self.stage_head = MitreStageClassifier(embedding_dim, self.num_classes)

        self.memory.load_state_dict(mem, strict=True)
        self.gnn.load_state_dict(ckpt['gnn'], strict=True)
        self.prob_head.load_state_dict(ckpt['prob_head'], strict=True)
        self.stage_head.load_state_dict(ckpt['mitre_head'], strict=True)

        self.current_flow_head = load_current_flow(ckpt)
        self.timestamp_encoding = ckpt.get("timestamp_encoding", "legacy_float32")
        for module in (self.memory, self.gnn, self.prob_head, self.stage_head):
            module.eval()

        # Allocate persistent components
        self.max_history = max_history
        self.neighbors = LastNeighborLoader(self.num_nodes, size=10, device='cpu')
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long)
        self.ip_mapper = PersistentIPMapper(self.num_nodes)  # headroom

        self.history_t = torch.empty(self.max_history, dtype=torch.long)
        self.history_msg = torch.empty((self.max_history, self.msg_dim), dtype=torch.float32)
        self.cursor = 0
        self.total_flows_scored = 0

        self.lock = threading.RLock()
        self.reset_state()

    def reset_state(self):
        """Flushes graph memory, neighbor connections, and IP maps."""
        with self.lock:
            self.memory.reset_state()
            self.neighbors.reset_state()
            self.ip_mapper.clear()
            self.cursor = 0
            self.total_flows_scored = 0

    @torch.no_grad()
    def _predict_and_update_batch(self, src: torch.Tensor, dst: torch.Tensor,
                                  t: torch.Tensor, msg: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Runs forward inference on a batch using current historical state, then appends to state."""
        batch_len = len(src)
        if self.cursor + batch_len > self.max_history:
            # When history buffer reaches capacity, compact / soft-reset edge buffer
            # (MemoryModule keeps node GRU representations)
            self.neighbors.reset_state()
            self.cursor = 0

        end = self.cursor + batch_len

        # GNN embedding pass
        unique_nodes = torch.cat([src, dst]).unique()
        nodes, edges, ids = self.neighbors(unique_nodes)
        self.assoc[nodes] = torch.arange(len(nodes))

        # Lookup edge messages and timestamps for neighbor aggregation
        edge_t = self.history_t[ids] if len(ids) > 0 else torch.empty(0, dtype=torch.long)
        edge_msg = self.history_msg[ids] if len(ids) > 0 else torch.empty((0, self.msg_dim), dtype=torch.float32)

        z = self.gnn(nodes, edges, edge_t, edge_msg)
        a, b = z[self.assoc[src]], z[self.assoc[dst]]

        attack_logits, stage_logits = combined_logits(self.prob_head, self.stage_head, self.current_flow_head, a, b, msg)
        probabilities = attack_logits.sigmoid().numpy()
        stages = stage_logits.softmax(-1).numpy()

        # Update persistent memory and neighbor graph
        self.memory.update_state(src, dst, t, msg)
        self.neighbors.insert(src, dst)

        self.history_t[self.cursor:end] = t
        self.history_msg[self.cursor:end] = msg
        self.cursor = end
        self.total_flows_scored += batch_len

        return probabilities, stages

    def score_flows(self, flows: List[Dict[str, Any]], feature_format: str = 'raw',
                    threshold: float = 0.5, batch_size: int = 200) -> List[Dict[str, Any]]:
        """Scores a list of flows while maintaining continuous temporal graph state."""
        if not flows:
            return []

        if not isinstance(batch_size, int) or not 1 <= batch_size <= 200:
            raise ValueError("batch_size must be between 1 and 200")
        if feature_format not in ("raw", "signed_log1p"):
            raise ValueError("Unknown feature format")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        times = np.asarray([f['timestamp'] for f in flows], dtype=np.float64)
        raw_feats = np.asarray([f['features'] for f in flows], dtype=np.float64)
        if raw_feats.shape != (len(flows), self.msg_dim) or not np.isfinite(raw_feats).all():
            raise ValueError(f"Each flow must contain {self.msg_dim} finite features")
        if not np.isfinite(times).all() or np.any(times < 0) or np.any(times > 1e12):
            raise ValueError("Timestamps must be finite and between 0 and 1e12")
        transformed_feats = (np.sign(raw_feats) * np.log1p(np.abs(raw_feats))
                             if feature_format == 'raw' else raw_feats)
        if np.any(np.abs(transformed_feats) > 100):
            raise ValueError("Transformed feature magnitude exceeds supported range (100)")
        # Bound batches so no endpoint can be recycled while still in use.
        batch_size = min(batch_size, self.max_history, max(1, self.ip_mapper.max_nodes // 2))
        results = []
        with self.lock:
            for start in range(0, len(flows), batch_size):
                end = min(start + batch_size, len(flows))
                chunk = flows[start:end]
                endpoints = {f[k] for f in chunk for k in ('src_ip', 'dst_ip')}
                if len(endpoints) > self.ip_mapper.max_nodes:
                    raise ValueError("Flow endpoints exceed node capacity")
                new_ips = endpoints.difference(self.ip_mapper.ip_to_id)
                if len(new_ips) > len(self.ip_mapper.free_ids):
                    # Recycling an ID without clearing all neighbor references leaks
                    # the previous host's state. Start a fresh graph generation.
                    total = self.total_flows_scored
                    self.reset_state()
                    self.total_flows_scored = total
                known = set(self.ip_mapper.ip_to_id)
                cold = [f['src_ip'] not in known and f['dst_ip'] not in known for f in chunk]
                src = torch.tensor([self.ip_mapper.get_or_create(f['src_ip'])[0] for f in chunk])
                dst = torch.tensor([self.ip_mapper.get_or_create(f['dst_ip'])[0] for f in chunk])
                t = encode_timestamps(times[start:end], self.timestamp_encoding)
                msg = torch.tensor(transformed_feats[start:end], dtype=torch.float32)
                p_batch, stages_batch = self._predict_and_update_batch(src, dst, t, msg)
                for offset, idx in enumerate(range(start, end)):
                    prob = float(p_batch[offset])
                    stage = int(stages_batch[offset].argmax())
                    flow_obj = flows[idx]
                    results.append({
                        "index": idx,
                        "src_ip": flow_obj['src_ip'],
                        "dst_ip": flow_obj['dst_ip'],
                        "timestamp": float(times[idx]),
                        "attack_probability": prob,
                        "is_attack": bool(prob >= threshold),
                        "mitre_stage": stage,
                        "stage_name": STAGES[stage] if stage < len(STAGES) else f'unmapped_{stage}',
                        "stage_probabilities": stages_batch[offset].tolist(),
                        "cold_start": cold[offset],
                        "metadata": flow_obj.get("metadata", {})
                    })
        return results
