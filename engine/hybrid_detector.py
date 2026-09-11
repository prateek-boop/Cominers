"""Dual Hybrid Threat Detection Engine.

Integrates:
1. Fast-Path Heuristic Engine (instant micro-second alerts on packet 1 for SYN/UDP floods, port scans, Slowloris)
2. CyberTGN Persistent Temporal Graph Network (deep graph memory for stealthy, complex attacks)
3. Confidence Arbiter fusing both signals into unified, high-accuracy threat verdicts.
"""
from __future__ import annotations
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any

from engine.streaming_runtime import StreamingModelRuntime


@dataclass
class HeuristicVerdict:
    is_threat: bool = False
    threat_type: str = "BENIGN"
    confidence: float = 0.0
    reason: str = "Normal traffic"


@dataclass
class UnifiedThreatAlert:
    index: int
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    protocol: str
    is_attack: bool
    final_score: float
    detection_source: str  # 'HEURISTIC_INSTANT', 'CYBERTGN_GRAPH', 'DUAL_CONFIRMED', or 'BENIGN'
    threat_type: str
    mitre_stage: str
    explanation: str
    cold_start: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


class FastHeuristicEngine:
    """Instant rule-based detection for obvious volumetric, handshake, and scan anomalies."""

    def __init__(self, scan_window_sec: float = 10.0, scan_port_threshold: int = 12):
        self.host_syn_history = defaultdict(deque)
        self.scan_window_sec = scan_window_sec
        self.scan_port_threshold = scan_port_threshold
        # Rolling history: src_ip -> deque of (dst_ip, dst_port, timestamp)
        self.host_target_history: Dict[str, deque] = defaultdict(deque)

    def _prune_history(self, src_ip: str, current_time: float):
        q = self.host_target_history[src_ip]
        cutoff = current_time - self.scan_window_sec
        while q and q[0][2] < cutoff:
            q.popleft()

    def evaluate_flow(self, flow: Dict[str, Any]) -> HeuristicVerdict:
        features = flow.get("features", [])
        if len(features) < 16:
            return HeuristicVerdict()

        src_ip = flow["src_ip"]
        dst_ip = flow["dst_ip"]
        t = flow.get("timestamp", time.time())
        meta = flow.get("metadata", {})
        protocol = meta.get("protocol", "tcp").lower()
        dst_port = meta.get("dst_port", 0)

        # Feature indices
        duration_us = features[0]
        fwd_pkts = features[1]
        bwd_pkts = features[2]
        tot_fwd_len = features[3]
        fwd_len_mean = features[5]
        flow_bytes_sec = features[7]
        flow_pkts_sec = features[8]
        flow_iat_mean = features[9]
        fwd_iat_mean = features[10]
        avg_pkt_size = features[14]
        down_up_ratio = features[15]

        # --- Rule 1: Port Scan Detection ---
        if dst_port > 0:
            self.host_target_history[src_ip].append((dst_ip, dst_port, t))
            self._prune_history(src_ip, t)
            distinct_ports = {entry[1] for entry in self.host_target_history[src_ip]}
            if len(distinct_ports) >= self.scan_port_threshold:
                return HeuristicVerdict(
                    is_threat=True,
                    threat_type="PORT_SCAN",
                    confidence=0.96,
                    reason=f"Host probed {len(distinct_ports)} distinct ports within {self.scan_window_sec}s window."
                )

        # Track SYN history across flows for multi-flow floods
        if protocol == "tcp" and meta.get("syn_count", fwd_pkts) > 0 and bwd_pkts == 0 and down_up_ratio == 0 and avg_pkt_size <= 70.0:
            if not hasattr(self, 'host_syn_history'):
                self.host_syn_history = defaultdict(deque)
            q_syn = self.host_syn_history[src_ip]
            q_syn.append(t)
            cutoff = t - self.scan_window_sec
            while q_syn and q_syn[0] < cutoff:
                q_syn.popleft()

            if fwd_pkts >= 3 or len(q_syn) >= 5:
                count = max(fwd_pkts, len(q_syn))
                conf = min(0.98, 0.75 + 0.04 * min(count, 6))
                return HeuristicVerdict(
                    is_threat=True,
                    threat_type="TCP_SYN_FLOOD",
                    confidence=conf,
                    reason=f"Unidirectional TCP SYN flood pattern ({count} SYNs observed, 0 responses, empty payload)."
                )

        # --- Rule 3: UDP Flood ---
        if protocol == "udp":
            if fwd_pkts >= 10 and bwd_pkts == 0 and flow_pkts_sec >= 100:
                return HeuristicVerdict(
                    is_threat=True,
                    threat_type="UDP_FLOOD",
                    confidence=0.93,
                    reason=f"High-rate unidirectional UDP blast ({flow_pkts_sec:.0f} pkts/s, 0 responses)."
                )

        # --- Rule 4: Slowloris DoS ---
        # Long duration, microscopic throughput, tiny forward packet lengths
        if protocol == "tcp" and duration_us > 15_000_000:  # > 15 seconds
            if fwd_len_mean <= 60 and flow_bytes_sec < 50.0 and fwd_iat_mean > 3_000_000:
                return HeuristicVerdict(
                    is_threat=True,
                    threat_type="SLOWLORIS_DOS",
                    confidence=0.91,
                    reason=f"Lingering TCP connection holding socket open (duration {duration_us/1e6:.1f}s, throughput {flow_bytes_sec:.1f} B/s)."
                )

        # --- Rule 5: Volumetric Exhaustion ---
        if fwd_pkts >= 20 and (flow_pkts_sec > 10_000 or flow_bytes_sec > 50_000_000):
            return HeuristicVerdict(
                is_threat=True,
                threat_type="VOLUMETRIC_DDoS",
                confidence=0.95,
                reason=f"Extreme volumetric surge detected ({flow_pkts_sec:.0f} pkts/s, {flow_bytes_sec/1e6:.1f} MB/s)."
            )

        return HeuristicVerdict()


class HybridDetector:
    """Combines FastHeuristicEngine with StreamingModelRuntime (CyberTGN)."""

    def __init__(self, checkpoint_path=None, threshold: float = 0.5):
        self.lock = threading.RLock()
        self.threshold = threshold
        self.heuristics = FastHeuristicEngine()
        if checkpoint_path:
            self.runtime = StreamingModelRuntime(checkpoint=checkpoint_path)
        else:
            self.runtime = StreamingModelRuntime()

    def reset_state(self):
        with self.lock:
            self.runtime.reset_state()
            self.heuristics.host_target_history.clear()
            self.heuristics.host_syn_history.clear()

    def analyze_flows(self, flows: List[Dict[str, Any]], feature_format: str = "raw") -> List[UnifiedThreatAlert]:
        """Runs dual-layer analysis across a stream or batch of flows."""
        if not flows:
            return []

        with self.lock:
            return self._analyze_flows(flows, feature_format)

    def _analyze_flows(self, flows, feature_format):
        # Validate and score before mutating heuristic histories.
        tgn_results = self.runtime.score_flows(
            flows,
            feature_format=feature_format,
            threshold=self.threshold,
            batch_size=min(200, len(flows))
        )

        if feature_format == "raw":
            heuristic_verdicts = [self.heuristics.evaluate_flow(f) for f in flows]
        else:
            import numpy as np
            heuristic_verdicts = [self.heuristics.evaluate_flow(dict(
                f, features=(np.sign(f["features"]) * np.expm1(np.abs(f["features"]))).tolist()
            )) for f in flows]

        unified_alerts: List[UnifiedThreatAlert] = []

        # 3. Fusion Arbiter
        for idx, (flow, h_verdict, tgn_res) in enumerate(zip(flows, heuristic_verdicts, tgn_results)):
            meta = flow.get("metadata", {})
            h_threat = h_verdict.is_threat
            tgn_threat = tgn_res["is_attack"]
            tgn_prob = tgn_res["attack_probability"]

            final_score = max(h_verdict.confidence if h_threat else 0.0, tgn_prob)
            is_attack = final_score >= self.threshold

            if h_threat and tgn_threat:
                source = "DUAL_CONFIRMED"
                threat_type = h_verdict.threat_type
                explanation = f"Confirmed by both Fast Heuristics ({h_verdict.reason}) and CyberTGN Graph Memory (Score {tgn_prob:.2f})."
            elif h_threat:
                source = "HEURISTIC_INSTANT"
                threat_type = h_verdict.threat_type
                explanation = f"Triggered by Fast-Path Heuristic: {h_verdict.reason}"
            elif tgn_threat:
                source = "CYBERTGN_GRAPH"
                threat_type = f"GRAPH_ANOMALY_{tgn_res['stage_name'].upper()}"
                explanation = f"Flagged by CyberTGN Temporal Graph (Score {tgn_prob:.2f}, Stage: {tgn_res['stage_name']})."
            else:
                source = "BENIGN"
                threat_type = "BENIGN"
                explanation = "Flow characteristics match normal baseline traffic."

            alert = UnifiedThreatAlert(
                index=idx,
                timestamp=flow["timestamp"],
                src_ip=flow["src_ip"],
                dst_ip=flow["dst_ip"],
                src_port=meta.get("src_port"),
                dst_port=meta.get("dst_port"),
                protocol=meta.get("protocol", "tcp"),
                is_attack=is_attack,
                final_score=float(final_score),
                detection_source=source,
                threat_type=threat_type,
                mitre_stage=tgn_res["stage_name"],
                explanation=explanation,
                cold_start=tgn_res["cold_start"],
                metadata=meta
            )
            unified_alerts.append(alert)

        return unified_alerts
