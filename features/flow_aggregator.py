"""Bidirectional Network Flow Aggregator.

Aggregates individual RawPacket streams into bidirectional flows over sliding time windows,
extracting both:
1. The 16 statistical features required by the CyberTGN checkpoint:
   - Flow Duration
   - Total Fwd Packet
   - Total Bwd packets
   - Total Length of Fwd Packet
   - Total Length of Bwd Packet
   - Fwd Packet Length Mean
   - Bwd Packet Length Mean
   - Flow Bytes/s
   - Flow Packets/s
   - Flow IAT Mean
   - Fwd IAT Mean
   - Bwd IAT Mean
   - Fwd Packets/s
   - Bwd Packets/s
   - Average Packet Size
   - Down/Up Ratio
2. Advanced threat metrics:
   - Shannon Payload Entropy
   - TCP Handshake Metrics (SYN-ACK ratio, latency)
   - Port Role Profile
"""
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from capture.sniffer import RawPacket
from .entropy import shannon_entropy, classify_entropy
from .handshake import HandshakeMetrics
from .port_roles import classify_port_roles, PortProfile


@dataclass
class FlowRecord:
    flow_id: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    start_time: float
    last_time: float

    # Raw packet collections
    fwd_packet_lengths: List[int] = field(default_factory=list)
    bwd_packet_lengths: List[int] = field(default_factory=list)
    fwd_timestamps: List[float] = field(default_factory=list)
    bwd_timestamps: List[float] = field(default_factory=list)
    all_timestamps: List[float] = field(default_factory=list)
    payload_buffer: bytearray = field(default_factory=bytearray)

    # Advanced Threat Modeling Metrics
    handshake: HandshakeMetrics = field(default_factory=HandshakeMetrics)
    port_profile: Optional[PortProfile] = None
    fin_fwd: bool = False
    fin_bwd: bool = False
    payload_entropy: float = 0.0
    entropy_class: str = "EMPTY"

    def add_packet(self, pkt: RawPacket):
        self.last_time = max(self.last_time, pkt.timestamp)
        self.all_timestamps.append(pkt.timestamp)

        is_forward = (pkt.src_ip == self.src_ip and pkt.src_port == self.src_port)

        if is_forward:
            self.fwd_packet_lengths.append(pkt.length)
            self.fwd_timestamps.append(pkt.timestamp)
        else:
            self.bwd_packet_lengths.append(pkt.length)
            self.bwd_timestamps.append(pkt.timestamp)

        if pkt.protocol == "tcp":
            if pkt.tcp_flags.get("FIN"):
                if is_forward:
                    self.fin_fwd = True
                else:
                    self.fin_bwd = True
            self.handshake.update_packet(is_forward, pkt.tcp_flags, pkt.timestamp)

        if pkt.payload:
            # Store up to 4KB of payload for entropy analysis
            if len(self.payload_buffer) < 4096:
                needed = 4096 - len(self.payload_buffer)
                self.payload_buffer.extend(pkt.payload[:needed])

    def finalize(self):
        """Compute threat modeling attributes and finalize flow stats."""
        self.port_profile = classify_port_roles(self.src_port, self.dst_port)
        if self.payload_buffer:
            self.payload_entropy = shannon_entropy(self.payload_buffer)
            self.entropy_class = classify_entropy(self.payload_entropy)
        else:
            self.payload_entropy = 0.0
            self.entropy_class = "EMPTY"

    def to_16_features(self) -> List[float]:
        """Produce the 16 numerical features expected by CyberTGN (CICFlowMeter convention)."""
        duration_sec = max(0.0, self.last_time - self.start_time)
        duration_usec = duration_sec * 1e6  # CICFlowMeter uses microseconds

        tot_fwd_pkts = len(self.fwd_packet_lengths)
        tot_bwd_pkts = len(self.bwd_packet_lengths)
        tot_pkts = tot_fwd_pkts + tot_bwd_pkts

        tot_fwd_len = sum(self.fwd_packet_lengths)
        tot_bwd_len = sum(self.bwd_packet_lengths)
        tot_bytes = tot_fwd_len + tot_bwd_len

        fwd_pkt_len_mean = float(np.mean(self.fwd_packet_lengths)) if tot_fwd_pkts > 0 else 0.0
        bwd_pkt_len_mean = float(np.mean(self.bwd_packet_lengths)) if tot_bwd_pkts > 0 else 0.0

        flow_bytes_per_sec = float(tot_bytes / duration_sec) if duration_sec > 0 else 0.0
        flow_pkts_per_sec = float(tot_pkts / duration_sec) if duration_sec > 0 else 0.0

        # Inter-Arrival Times (IAT) in microseconds
        def _calc_iat_mean(ts_list: List[float]) -> float:
            if len(ts_list) <= 1:
                return 0.0
            sorted_ts = sorted(ts_list)
            diffs = np.diff(sorted_ts) * 1e6
            return float(np.mean(diffs)) if len(diffs) > 0 else 0.0

        flow_iat_mean = _calc_iat_mean(self.all_timestamps)
        fwd_iat_mean = _calc_iat_mean(self.fwd_timestamps)
        bwd_iat_mean = _calc_iat_mean(self.bwd_timestamps)

        fwd_pkts_per_sec = float(tot_fwd_pkts / duration_sec) if duration_sec > 0 else 0.0
        bwd_pkts_per_sec = float(tot_bwd_pkts / duration_sec) if duration_sec > 0 else 0.0

        avg_pkt_size = float(tot_bytes / max(1, tot_pkts))
        down_up_ratio = float(tot_bwd_pkts / max(1, tot_fwd_pkts))

        features = [
            float(duration_usec),
            float(tot_fwd_pkts),
            float(tot_bwd_pkts),
            float(tot_fwd_len),
            float(tot_bwd_len),
            float(fwd_pkt_len_mean),
            float(bwd_pkt_len_mean),
            float(flow_bytes_per_sec),
            float(flow_pkts_per_sec),
            float(flow_iat_mean),
            float(fwd_iat_mean),
            float(bwd_iat_mean),
            float(fwd_pkts_per_sec),
            float(bwd_pkts_per_sec),
            float(avg_pkt_size),
            float(down_up_ratio),
        ]
        return features


class FlowAggregator:
    """Aggregates packets into active flows and flushes expired flows based on inactivity timeout."""

    def __init__(self, flow_timeout_sec: float = 10.0, max_active_flows: int = 10000, active_timeout_sec: float = 120.0):
        if max_active_flows < 1 or flow_timeout_sec <= 0:
            raise ValueError("Flow capacity and timeout must be positive")
        self.active_timeout_sec = active_timeout_sec
        self._pending = []
        self.flow_timeout_sec = flow_timeout_sec
        self.max_active_flows = max_active_flows
        self.active_flows: Dict[str, FlowRecord] = {}

    def _get_flow_key(self, pkt: RawPacket) -> Tuple[str, bool]:
        """Compute canonical flow key and direction."""
        fwd_tuple = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
        bwd_tuple = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)

        fwd_key = f"{fwd_tuple[0]}:{fwd_tuple[2]}->{fwd_tuple[1]}:{fwd_tuple[3]}/{fwd_tuple[4]}"
        bwd_key = f"{bwd_tuple[0]}:{bwd_tuple[2]}->{bwd_tuple[1]}:{bwd_tuple[3]}/{bwd_tuple[4]}"

        if fwd_key in self.active_flows:
            return fwd_key, True
        elif bwd_key in self.active_flows:
            return bwd_key, False
        else:
            return fwd_key, True

    def add_packet(self, pkt: RawPacket) -> Optional[FlowRecord]:
        """Add packet to active flow. Returns flow if terminated by FIN/RST."""
        key, is_forward = self._get_flow_key(pkt)
        evicted = None

        if key not in self.active_flows:
            if len(self.active_flows) >= self.max_active_flows:
                # Evict oldest flow if capacity reached
                oldest_key = min(self.active_flows.keys(), key=lambda k: self.active_flows[k].last_time)
                evicted = self.active_flows.pop(oldest_key)
                evicted.finalize()

            self.active_flows[key] = FlowRecord(
                flow_id=key,
                src_ip=pkt.src_ip,
                dst_ip=pkt.dst_ip,
                src_port=pkt.src_port,
                dst_port=pkt.dst_port,
                protocol=pkt.protocol,
                start_time=pkt.timestamp,
                last_time=pkt.timestamp,
            )

        flow = self.active_flows[key]
        flow.add_packet(pkt)

        # If TCP connection was closed via RST or FIN-ACK, immediately flush
        if pkt.protocol == "tcp" and (pkt.tcp_flags.get("RST") or (flow.fin_fwd and flow.fin_bwd)):
            flow.finalize()
            del self.active_flows[key]
            if evicted is not None:
                self._pending.append(evicted)
            return flow

        return evicted

    def flush_expired(self, current_time: float) -> List[FlowRecord]:
        """Flush flows that have been idle longer than flow_timeout_sec."""
        expired, self._pending = self._pending, []
        expired_keys = []
        for key, flow in self.active_flows.items():
            if current_time - flow.last_time >= self.flow_timeout_sec or current_time - flow.start_time >= self.active_timeout_sec:
                flow.finalize()
                expired.append(flow)
                expired_keys.append(key)

        for key in expired_keys:
            del self.active_flows[key]

        return expired

    def flush_all(self) -> List[FlowRecord]:
        """Flush all active flows unconditionally."""
        flushed, self._pending = self._pending, []
        for flow in self.active_flows.values():
            flow.finalize()
            flushed.append(flow)
        self.active_flows.clear()
        return flushed
