"""Stateful Streaming CyberTGN Engine with Conformal Bound.

Maintains persistent temporal graph memory across sliding streaming windows,
scores flows dynamically using CyberTGN, verifies alerts with a Conformal Bound,
and decodes MITRE ATT&CK stages.
"""
from dataclasses import dataclass, field
from pathlib import Path
import threading
import uuid
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.nn.models.tgn import LastNeighborLoader

from models.current_flow import load_current_flow, combined_logits, encode_timestamps
from models.memory import MemoryModule
from models.tgn import CyberTGN
from models.multitask_heads import AttackProbabilityHead, MitreStageClassifier
from graph.graph_formatter import FormattedGraphBatch
from .conformal import SplitConformalPredictor

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHECKPOINT = ROOT / "checkpoints/tgn_best.pt"

MITRE_STAGES = [
    "benign",
    "reconnaissance",
    "initial_access",
    "credential_access",
    "lateral_movement",
    "command_and_control",
    "exfiltration",
    "impact",
]


@dataclass
class ThreatAlert:
    alert_id: str
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    attack_probability: float
    conformal_p_value: float
    conformal_confirmed: bool
    mitre_stage: int
    stage_name: str
    payload_entropy: float
    syn_ack_ratio: float
    features_16: List[float]
    flow_id: str
    extra: Dict = field(default_factory=dict)


class StatefulCyberTGN:
    """Persistent stateful CyberTGN inference engine for streaming packet/flow graphs."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        history_capacity: int = 50000,
        conformal_alpha: float = 0.05,
        alert_threshold: float = 0.85,
        capture_explanation_context: bool = False,
    ):
        self.checkpoint_path = Path(checkpoint_path).resolve()
        if history_capacity < 1:
            raise ValueError("history_capacity must be positive")
        self.history_capacity = history_capacity
        self.alert_threshold = alert_threshold
        self.lock = threading.Lock()
        self.capture_explanation_context = capture_explanation_context
        self.last_explanation_context = None

        # 1. Load checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.stage_validation_passed = ckpt.get('stage_validation', {}).get('targets_met') is True
        mem = ckpt["memory"]
        self.num_nodes, dim = mem["memory.memory"].shape
        time_dim = mem["memory.time_enc.lin.weight"].shape[0]
        self.msg_dim = mem["memory.gru.weight_ih"].shape[1] - 2 * dim - time_dim
        embedding_dim = ckpt["prob_head"]["mlp.0.weight"].shape[1] // 2
        self.num_classes = ckpt["mitre_head"]["mlp.2.weight"].shape[0]

        # 2. Instantiate modules
        time_transform = ckpt.get('model_config',{}).get('time_transform','identity')
        self.memory = MemoryModule(self.num_nodes, self.msg_dim, dim, time_dim,time_transform)
        self.gnn = CyberTGN(self.memory, dim, embedding_dim, self.msg_dim, time_dim,time_transform)
        self.prob_head = AttackProbabilityHead(embedding_dim)
        self.stage_head = MitreStageClassifier(embedding_dim, self.num_classes)

        # 3. Load weights strictly
        self.memory.load_state_dict(mem, strict=True)
        self.gnn.load_state_dict(ckpt["gnn"], strict=True)
        self.prob_head.load_state_dict(ckpt["prob_head"], strict=True)
        self.stage_head.load_state_dict(ckpt["mitre_head"], strict=True)

        self.current_flow_head = load_current_flow(ckpt)
        self.timestamp_encoding = ckpt.get("timestamp_encoding", "legacy_float32")
        for module in (self.memory, self.gnn, self.prob_head, self.stage_head):
            module.eval()

        # Checkpoint memory belongs to training hosts, not this deployment.
        self.memory.reset_state()

        # 4. Neighbor loader and history buffers
        self.neighbors = LastNeighborLoader(self.num_nodes, size=10, device="cpu")
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long)

        # Pre-allocate circular history tensors
        self.history_t = torch.zeros(self.history_capacity, dtype=torch.long)
        self.history_msg = torch.zeros((self.history_capacity, self.msg_dim), dtype=torch.float32)
        self.cursor = 0
        self.total_events_seen = 0

        # 5. Conformal Predictor
        self.conformal = SplitConformalPredictor(alpha=conformal_alpha)

    def process_graph_batch(
        self,
        batch: FormattedGraphBatch,
    ) -> Tuple[List[dict], List[ThreatAlert]]:
        """
        Process a dynamic graph batch through the stateful TGN:
        1. Query neighbors and run GNN forward pass.
        2. Score attack probability and MITRE ATT&CK stage logits.
        3. Evaluate conformal bound.
        4. Update node memories and neighbor graph.
        5. Generate ThreatAlerts for high-risk flows.
        """
        src = batch.src_nodes
        dst = batch.dst_nodes
        t = (encode_timestamps([f.last_time for f in batch.flows], self.timestamp_encoding)
             if self.timestamp_encoding == "int64_seconds" else batch.timestamps)
        msg = batch.edge_features
        flows = batch.flows
        batch_len = len(src)

        if batch_len == 0:
            return [], []

        if batch_len > self.history_capacity:
            raise ValueError("Batch exceeds history capacity; split into smaller batches")

        with self.lock:
            with torch.no_grad():
                if self.cursor + batch_len > self.history_capacity:
                    self.neighbors.reset_state()
                    self.cursor = 0
                # 1. Fetch temporal 1-hop / 2-hop neighborhood
                nodes, edges, ids = self.neighbors(torch.cat([src, dst]).unique())
                self.assoc[nodes] = torch.arange(len(nodes))

                # Safe indexing into history tensors
                sub_t = self.history_t[ids] if len(ids) > 0 else torch.empty(0, dtype=torch.long)
                sub_msg = self.history_msg[ids] if len(ids) > 0 else torch.empty((0, self.msg_dim), dtype=torch.float32)

                # 2. GNN forward pass
                if self.capture_explanation_context:
                    from explainability.model_explainer import FrozenPrediction
                    self.last_explanation_context = FrozenPrediction(
                        self, nodes, edges, sub_t, sub_msg, self.assoc[src], self.assoc[dst], msg)
                z = self.gnn(nodes, edges, sub_t, sub_msg)
                z_src, z_dst = z[self.assoc[src]], z[self.assoc[dst]]

                # 3. Heads
                attack_logits, stage_logits = combined_logits(self.prob_head, self.stage_head, self.current_flow_head, z_src, z_dst, msg)
                probabilities = attack_logits.sigmoid().numpy()
                stages_logits = stage_logits.softmax(-1).numpy()

                # 4. Update memory and neighbors with the new batch interactions
                self.memory.update_state(src, dst, t.long(), msg)
                self.neighbors.insert(src, dst)

                # Neighbor event IDs and history offsets must stay aligned.
                end = self.cursor + batch_len
                self.history_t[self.cursor:end] = t.long()
                self.history_msg[self.cursor:end] = msg
                self.cursor = end

                self.total_events_seen += batch_len

        results = []
        alerts = []

        for i, flow in enumerate(flows):
            p_gnn = float(probabilities[i]) if probabilities.ndim > 0 else float(probabilities)
            stage_idx = int(stages_logits[i].argmax()) if stages_logits.ndim > 1 else int(stages_logits.argmax())

            # Threat Modeling Ground-Truthing (Layer 2)
            p_threat = 0.0
            threat_stage = None

            # High payload entropy (encrypted exploit payload / shellcode / packed C2)
            if flow.payload_entropy >= 7.0 and len(flow.payload_buffer) >= 64:
                p_threat = max(p_threat, 0.92)
                threat_stage = "initial_access"

            # Abnormal SYN-ACK ratio (SYN flood / stealth scan)
            if flow.handshake.syn_ack_ratio >= 3.0 and flow.handshake.syn_count >= 5:
                p_threat = max(p_threat, 0.89)
                threat_stage = "reconnaissance"

            # Combine GNN score with Layer 2 Threat Modeling ground-truthed score
            p_attack = max(p_gnn, p_threat)
            if threat_stage and p_threat > p_gnn:
                stage_name = threat_stage
                stage_idx = MITRE_STAGES.index(threat_stage) if threat_stage in MITRE_STAGES else stage_idx
            else:
                stage_name = MITRE_STAGES[stage_idx] if stage_idx < len(MITRE_STAGES) else f"stage_{stage_idx}"

            # Conformal assessment
            conf_eval = self.conformal.evaluate_flow(p_attack, base_threshold=self.alert_threshold)

            record = {
                "flow_id": flow.flow_id,
                "src_ip": flow.src_ip,
                "dst_ip": flow.dst_ip,
                "src_port": flow.src_port,
                "dst_port": flow.dst_port,
                "protocol": flow.protocol,
                "timestamp": flow.last_time,
                "attack_probability": p_attack,
                "model_attack_probability": p_gnn,
                "heuristic_probability": p_threat,
                "stage_probabilities": stages_logits[i].tolist(),
                "mitre_stage": stage_idx,
                "stage_name": stage_name,
                "conformal_p_value": conf_eval["conformal_p_value"],
                "conformal_confirmed": conf_eval["conformal_confirmed"],
                "is_high_risk": conf_eval["is_high_risk_alert"],
                "payload_entropy": flow.payload_entropy,
                "syn_ack_ratio": flow.handshake.syn_ack_ratio,
            }
            results.append(record)

            # Check if condition for High-Risk Infiltration Alert: >85% + Conformal confirmation
            if conf_eval["is_high_risk_alert"]:
                alert = ThreatAlert(
                    alert_id=str(uuid.uuid4()),
                    timestamp=flow.last_time,
                    src_ip=flow.src_ip,
                    dst_ip=flow.dst_ip,
                    src_port=flow.src_port,
                    dst_port=flow.dst_port,
                    protocol=flow.protocol,
                    attack_probability=p_attack,
                    conformal_p_value=conf_eval["conformal_p_value"],
                    conformal_confirmed=conf_eval["conformal_confirmed"],
                    mitre_stage=stage_idx,
                    stage_name=stage_name,
                    payload_entropy=flow.payload_entropy,
                    syn_ack_ratio=flow.handshake.syn_ack_ratio,
                    features_16=flow.to_16_features(),
                    flow_id=flow.flow_id,
                    extra={
                        "model_attack_probability": p_gnn,
                        "stage_validation_passed": self.stage_validation_passed,
                        "stage_probabilities": stages_logits[i].tolist(),
                        "flow_start_time": flow.start_time,
                        "calibration_source": "provided_benign_scores" if self.conformal.is_calibrated else "uncalibrated",
                        "port_profile": flow.port_profile.__dict__ if flow.port_profile else {},
                        "handshake": flow.handshake.__dict__,
                    },
                )
                alerts.append(alert)

        return results, alerts
