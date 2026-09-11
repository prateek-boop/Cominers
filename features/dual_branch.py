"""Phase 1: Dual-Branch Feature Engineering.

Extracts metrics across two complementary analytical branches:
1. Packet-Level (Micro dynamics):
   - TTL variance
   - TCP Window size variance
   - TCP Flag ratios (SYN, ACK, RST, FIN, PSH)
   - Payload byte Shannon entropy
2. Flow-Level (Macro dynamics):
   - Flow Duration
   - Inter-Arrival Time (IAT Mean, IAT StdDev)
   - Forward/Backward Packet Length statistics
   - Flow Bytes/sec and Packets/sec

Forms the 32-dimensional Edge Feature Vector (f_e in R^32) and Node Features (x_v in R^16).
"""
from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from capture.sniffer import RawPacket
from .entropy import shannon_entropy


@dataclass
class MicroPacketMetrics:
    ttl_values: List[int] = field(default_factory=list)
    window_sizes: List[int] = field(default_factory=list)
    syn_count: int = 0
    ack_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    psh_count: int = 0
    payload_entropy: float = 0.0

    def compute_vector_16(self) -> np.ndarray:
        """Produce 16-dimensional Micro-dynamic packet feature vector."""
        ttl_var = float(np.var(self.ttl_values)) if len(self.ttl_values) > 1 else 0.0
        ttl_mean = float(np.mean(self.ttl_values)) if self.ttl_values else 64.0
        win_var = float(np.var(self.window_sizes)) if len(self.window_sizes) > 1 else 0.0
        win_mean = float(np.mean(self.window_sizes)) if self.window_sizes else 0.0

        total_flags = max(1, self.syn_count + self.ack_count + self.rst_count + self.fin_count + self.psh_count)
        syn_ratio = float(self.syn_count / total_flags)
        ack_ratio = float(self.ack_count / total_flags)
        rst_ratio = float(self.rst_count / total_flags)
        fin_ratio = float(self.fin_count / total_flags)
        psh_ratio = float(self.psh_count / total_flags)
        syn_ack_ratio = float(self.syn_count / (self.ack_count + 1))

        return np.array([
            ttl_mean, ttl_var, win_mean, win_var,
            float(self.syn_count), float(self.ack_count), float(self.rst_count),
            float(self.fin_count), float(self.psh_count),
            syn_ratio, ack_ratio, rst_ratio, fin_ratio, psh_ratio,
            syn_ack_ratio, float(self.payload_entropy)
        ], dtype=np.float32)


@dataclass
class MacroFlowMetrics:
    duration_sec: float = 0.0
    fwd_lengths: List[int] = field(default_factory=list)
    bwd_lengths: List[int] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    fwd_timestamps: List[float] = field(default_factory=list)
    bwd_timestamps: List[float] = field(default_factory=list)

    def compute_vector_16(self) -> np.ndarray:
        """Produce 16-dimensional Macro-dynamic flow feature vector."""
        dur = max(1e-6, self.duration_sec)
        tot_fwd = len(self.fwd_lengths)
        tot_bwd = len(self.bwd_lengths)
        tot_pkts = tot_fwd + tot_bwd
        tot_bytes = sum(self.fwd_lengths) + sum(self.bwd_lengths)

        fwd_mean = float(np.mean(self.fwd_lengths)) if tot_fwd > 0 else 0.0
        bwd_mean = float(np.mean(self.bwd_lengths)) if tot_bwd > 0 else 0.0

        bytes_per_sec = float(tot_bytes / dur)
        pkts_per_sec = float(tot_pkts / dur)

        def _calc_iat(ts_list):
            if len(ts_list) <= 1:
                return 0.0, 0.0
            diffs = np.diff(sorted(ts_list))
            return float(np.mean(diffs)), float(np.std(diffs))

        iat_mean, iat_std = _calc_iat(self.timestamps)
        fwd_iat_mean, fwd_iat_std = _calc_iat(self.fwd_timestamps)
        bwd_iat_mean, bwd_iat_std = _calc_iat(self.bwd_timestamps)

        down_up_ratio = float(tot_bwd / max(1, tot_fwd))
        avg_pkt_size = float(tot_bytes / max(1, tot_pkts))

        return np.array([
            float(dur), float(tot_fwd), float(tot_bwd),
            fwd_mean, bwd_mean, bytes_per_sec, pkts_per_sec,
            iat_mean, iat_std, fwd_iat_mean, fwd_iat_std, bwd_iat_mean, bwd_iat_std,
            down_up_ratio, avg_pkt_size, float(tot_bytes)
        ], dtype=np.float32)


class DualBranchFeatureExtractor:
    """Combines packet micro-dynamics and flow macro-dynamics into unified representations."""

    @staticmethod
    def extract_edge_feature_vector_32(micro: MicroPacketMetrics, macro: MacroFlowMetrics) -> np.ndarray:
        """Combine micro and macro feature vectors into Phase 2 Edge Vector f_e in R^32."""
        vec_micro = micro.compute_vector_16()
        vec_macro = macro.compute_vector_16()
        return np.concatenate([vec_micro, vec_macro])

    @staticmethod
    def extract_node_feature_vector_16(
        in_degree: int,
        out_degree: int,
        internal_external_ratio: float,
        packet_volume_delta: float,
        role_id: int,
        additional_metrics: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Form Phase 2 Node Feature Vector x_v in R^16."""
        base = [
            float(in_degree),
            float(out_degree),
            float(in_degree + out_degree),
            float(in_degree / max(1, out_degree)),
            float(internal_external_ratio),
            float(packet_volume_delta),
            float(role_id),
            float(1.0 if role_id == 0 else 0.0),  # is_client
            float(1.0 if role_id == 1 else 0.0),  # is_server
            float(1.0 if role_id == 2 else 0.0),  # is_admin_or_core
            float(1.0 if role_id == 3 else 0.0),  # is_sensitive_telecom_calea
        ]
        # Pad or slice to exactly 16 dimensions
        remainder = 16 - len(base)
        if additional_metrics:
            base.extend(additional_metrics[:remainder])
        while len(base) < 16:
            base.append(0.0)

        return np.array(base[:16], dtype=np.float32)
