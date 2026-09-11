"""Real-time and PCAP Bidirectional Flow Engine.

Assembles packets into bidirectional flows, tracks TCP handshake state,
and extracts the exact 16 CICFlowMeter statistical features required by CyberTGN.
"""
from __future__ import annotations
import math
import threading
from functools import wraps

def synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapped

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Iterator, Any
import numpy as np

# Feature order strictly matching model_runtime.py
FEATURE_NAMES = [
    'Flow Duration',
    'Total Fwd Packet',
    'Total Bwd packets',
    'Total Length of Fwd Packet',
    'Total Length of Bwd Packet',
    'Fwd Packet Length Mean',
    'Bwd Packet Length Mean',
    'Flow Bytes/s',
    'Flow Packets/s',
    'Flow IAT Mean',
    'Fwd IAT Mean',
    'Bwd IAT Mean',
    'Fwd Packets/s',
    'Bwd Packets/s',
    'Average Packet Size',
    'Down/Up Ratio',
]


@dataclass
class RawPacketInfo:
    timestamp: float  # Unix epoch in seconds (float)
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str  # 'tcp', 'udp', or 'other'
    length: int
    tcp_flags: Optional[int] = None  # Bitmask if TCP (SYN=0x02, ACK=0x10, FIN=0x01, RST=0x04, PSH=0x08)
    tcp_window: Optional[int] = None
    ttl: Optional[int] = None


@dataclass
class BidirFlow:
    flow_id: Tuple[str, str, int, int, str]  # (fwd_src_ip, fwd_dst_ip, fwd_src_port, fwd_dst_port, protocol)
    fwd_src_ip: str
    fwd_dst_ip: str
    fwd_src_port: int
    fwd_dst_port: int
    protocol: str

    start_time: float
    last_time: float

    # Forward direction
    fwd_packet_count: int = 0
    fwd_byte_count: int = 0
    fwd_packet_lengths: List[int] = field(default_factory=list)
    fwd_timestamps: List[float] = field(default_factory=list)
    fwd_tcp_flags: List[int] = field(default_factory=list)

    # Backward direction
    bwd_packet_count: int = 0
    bwd_byte_count: int = 0
    bwd_packet_lengths: List[int] = field(default_factory=list)
    bwd_timestamps: List[float] = field(default_factory=list)
    bwd_tcp_flags: List[int] = field(default_factory=list)

    # Combined timestamps in arrival order
    flow_timestamps: List[float] = field(default_factory=list)

    # Handshake / State Tracking
    syn_fwd: bool = False
    syn_ack_bwd: bool = False
    ack_fwd: bool = False
    is_handshake_complete: bool = False
    fin_fwd: bool = False
    fin_bwd: bool = False
    rst_seen: bool = False

    def add_packet(self, pkt: RawPacketInfo, is_forward: bool):
        t = pkt.timestamp
        self.last_time = max(self.last_time, t)
        self.flow_timestamps.append(t)

        flags = pkt.tcp_flags or 0
        is_syn = bool(flags & 0x02)
        is_ack = bool(flags & 0x10)
        is_rst = bool(flags & 0x04)
        is_fin = bool(flags & 0x01)

        if is_rst:
            self.rst_seen = True

        if is_forward:
            self.fwd_packet_count += 1
            self.fwd_byte_count += pkt.length
            self.fwd_packet_lengths.append(pkt.length)
            self.fwd_timestamps.append(t)
            if pkt.tcp_flags is not None:
                self.fwd_tcp_flags.append(pkt.tcp_flags)
                if is_syn and not is_ack:
                    self.syn_fwd = True
                if self.syn_fwd and self.syn_ack_bwd and is_ack:
                    self.ack_fwd = True
                    self.is_handshake_complete = True
                if is_fin:
                    self.fin_fwd = True
        else:
            self.bwd_packet_count += 1
            self.bwd_byte_count += pkt.length
            self.bwd_packet_lengths.append(pkt.length)
            self.bwd_timestamps.append(t)
            if pkt.tcp_flags is not None:
                self.bwd_tcp_flags.append(pkt.tcp_flags)
                if is_syn and is_ack:
                    self.syn_ack_bwd = True
                if is_fin:
                    self.fin_bwd = True

    def is_terminated(self) -> bool:
        """Returns True if connection closed via RST or dual FIN."""
        if self.rst_seen:
            return True
        if self.fin_fwd and self.fin_bwd:
            return True
        return False

    def calculate_iat_mean(self, timestamps: List[float]) -> float:
        """Returns mean Inter-Arrival Time in microseconds."""
        if len(timestamps) < 2:
            return 0.0
        # Calculate consecutive differences in microseconds
        deltas = [(timestamps[i] - timestamps[i - 1]) * 1e6 for i in range(1, len(timestamps))]
        # Exclude negative deltas from out-of-order packets if any
        positive_deltas = [d for d in deltas if d >= 0]
        if not positive_deltas:
            return 0.0
        return float(np.mean(positive_deltas))

    def to_features(self) -> List[float]:
        """Extracts exactly the 16 features expected by CyberTGN in CICFlowMeter units."""
        # 1. Flow Duration in microseconds
        duration_sec = max(self.last_time - self.start_time, 0.0)
        duration_us = duration_sec * 1e6

        # Effective duration for rate calculations (avoid zero division)
        eff_duration_sec = max(duration_sec, 1e-6)

        # 2-5: Packet and Byte Counts
        fwd_pkts = self.fwd_packet_count
        bwd_pkts = self.bwd_packet_count
        tot_fwd_len = float(self.fwd_byte_count)
        tot_bwd_len = float(self.bwd_byte_count)

        # 6-7: Packet Length Means
        fwd_len_mean = float(np.mean(self.fwd_packet_lengths)) if self.fwd_packet_lengths else 0.0
        bwd_len_mean = float(np.mean(self.bwd_packet_lengths)) if self.bwd_packet_lengths else 0.0

        # 8-9: Flow Rates (0 if duration is 0 to avoid microsecond skew)
        tot_bytes = tot_fwd_len + tot_bwd_len
        tot_pkts = fwd_pkts + bwd_pkts
        if duration_sec > 0:
            flow_bytes_per_sec = tot_bytes / duration_sec
            flow_pkts_per_sec = tot_pkts / duration_sec
            fwd_pkts_per_sec = fwd_pkts / duration_sec
            bwd_pkts_per_sec = bwd_pkts / duration_sec
        else:
            flow_bytes_per_sec = 0.0
            flow_pkts_per_sec = 0.0
            fwd_pkts_per_sec = 0.0
            bwd_pkts_per_sec = 0.0

        # 10-12: Inter-Arrival Times (in microseconds)
        flow_iat_mean = self.calculate_iat_mean(self.flow_timestamps)
        fwd_iat_mean = self.calculate_iat_mean(self.fwd_timestamps)
        bwd_iat_mean = self.calculate_iat_mean(self.bwd_timestamps)

        # 15: Average Packet Size
        avg_pkt_size = (tot_bytes / tot_pkts) if tot_pkts > 0 else 0.0

        # 16: Down/Up Ratio (Bwd / Fwd)
        down_up_ratio = float(bwd_pkts / fwd_pkts) if fwd_pkts > 0 else 0.0

        features = [
            float(duration_us),
            float(fwd_pkts),
            float(bwd_pkts),
            float(tot_fwd_len),
            float(tot_bwd_len),
            float(fwd_len_mean),
            float(bwd_len_mean),
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

    def to_flow_dict(self) -> Dict[str, Any]:
        """Formats the flow record for model input."""
        return {
            "src_ip": self.fwd_src_ip,
            "dst_ip": self.fwd_dst_ip,
            "timestamp": float(self.start_time),
            "features": self.to_features(),
            "metadata": {
                "src_port": self.fwd_src_port,
                "dst_port": self.fwd_dst_port,
                "protocol": self.protocol,
                "duration_sec": self.last_time - self.start_time,
                "fwd_packets": self.fwd_packet_count,
                "bwd_packets": self.bwd_packet_count,
                "syn_count": sum(bool(flags & 2) and not bool(flags & 16) for flags in self.fwd_tcp_flags),
                "handshake_complete": self.is_handshake_complete,
                "rst_seen": self.rst_seen,
            },
        }


class FlowEngine:
    """Manages active bidirectional flows and exports them upon completion or timeout."""

    def __init__(self, idle_timeout_sec: float = 15.0, active_timeout_sec: float = 120.0, max_active_flows: int = 10000):
        if min(idle_timeout_sec, active_timeout_sec, max_active_flows) <= 0:
            raise ValueError("Timeouts and flow capacity must be positive")
        self.max_active_flows = max_active_flows
        self.lock = threading.RLock()
        self.idle_timeout_sec = idle_timeout_sec
        self.active_timeout_sec = active_timeout_sec
        self.active_flows: Dict[Tuple[str, str, int, int, str], BidirFlow] = {}
        self.total_packets_processed = 0
        self.total_flows_flushed = 0

    def _get_key(self, pkt: RawPacketInfo) -> Tuple[Tuple[str, str, int, int, str], bool]:
        """Canonical key lookup. Returns (canonical_key, is_forward)."""
        fwd_key = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
        bwd_key = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)

        if fwd_key in self.active_flows:
            return fwd_key, True
        if bwd_key in self.active_flows:
            return bwd_key, False

        # If not existing, create new forward flow with src as initiator
        return fwd_key, True

    @synchronized
    def process_packet(self, pkt: RawPacketInfo) -> List[Dict[str, Any]]:
        """Processes a single packet. Returns any completed flows ready for scoring."""
        self.total_packets_processed += 1
        completed = self.flush_expired(pkt.timestamp)
        key, is_forward = self._get_key(pkt)

        if key not in self.active_flows:
            if len(self.active_flows) >= self.max_active_flows:
                oldest = min(self.active_flows, key=lambda k: self.active_flows[k].last_time)
                completed.append(self.active_flows.pop(oldest).to_flow_dict())
                self.total_flows_flushed += 1
            flow = BidirFlow(
                flow_id=key,
                fwd_src_ip=key[0],
                fwd_dst_ip=key[1],
                fwd_src_port=key[2],
                fwd_dst_port=key[3],
                protocol=key[4],
                start_time=pkt.timestamp,
                last_time=pkt.timestamp,
            )
            self.active_flows[key] = flow
        else:
            flow = self.active_flows[key]

        flow.add_packet(pkt, is_forward)

        # Check if flow terminated (RST or FIN-FIN)
        if flow.is_terminated():
            completed.append(flow.to_flow_dict())
            del self.active_flows[key]
            self.total_flows_flushed += 1

        return completed

    @synchronized
    def flush_expired(self, current_time: float) -> List[Dict[str, Any]]:
        """Flushes flows that exceeded idle or active timeout."""
        expired_keys = []
        flushed = []
        for key, flow in self.active_flows.items():
            idle_time = current_time - flow.last_time
            active_time = current_time - flow.start_time
            if idle_time >= self.idle_timeout_sec or active_time >= self.active_timeout_sec:
                flushed.append(flow.to_flow_dict())
                expired_keys.append(key)

        for key in expired_keys:
            del self.active_flows[key]
        self.total_flows_flushed += len(expired_keys)

        return flushed

    @synchronized
    def flush_all(self) -> List[Dict[str, Any]]:
        """Forces all active flows to export."""
        flushed = [f.to_flow_dict() for f in self.active_flows.values()]
        self.total_flows_flushed += len(flushed)
        self.active_flows.clear()
        return flushed

    @staticmethod
    def parse_scapy_packet(packet) -> Optional[RawPacketInfo]:
        """Converts a Scapy packet object into RawPacketInfo."""
        try:
            # Check IP layer (IPv4)
            if not hasattr(packet, 'haslayer'):
                return None

            from scapy.layers.inet import IP, TCP, UDP
            if not packet.haslayer(IP):
                return None

            ip_layer = packet[IP]
            src_ip = ip_layer.src
            dst_ip = ip_layer.dst
            pkt_len = len(packet)
            pkt_time = float(packet.time)
            ttl = int(ip_layer.ttl) if hasattr(ip_layer, 'ttl') else None

            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                return RawPacketInfo(
                    timestamp=pkt_time,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=int(tcp_layer.sport),
                    dst_port=int(tcp_layer.dport),
                    protocol="tcp",
                    length=pkt_len,
                    tcp_flags=int(tcp_layer.flags),
                    tcp_window=int(tcp_layer.window),
                    ttl=ttl,
                )
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                return RawPacketInfo(
                    timestamp=pkt_time,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=int(udp_layer.sport),
                    dst_port=int(udp_layer.dport),
                    protocol="udp",
                    length=pkt_len,
                    tcp_flags=None,
                    tcp_window=None,
                    ttl=ttl,
                )
            else:
                return RawPacketInfo(
                    timestamp=pkt_time,
                    src_ip=src_ip,
                    dst_ip=dst_ip,
                    src_port=0,
                    dst_port=0,
                    protocol="other",
                    length=pkt_len,
                    tcp_flags=None,
                    tcp_window=None,
                    ttl=ttl,
                )
        except Exception:
            return None

    def process_pcap_file(self, pcap_path: str) -> List[Dict[str, Any]]:
        """Reads a PCAP file using Scapy PcapReader and returns all extracted flows."""
        from scapy.utils import PcapReader
        flows: List[Dict[str, Any]] = []
        last_time = 0.0

        with open(pcap_path, "rb") as capture_file, PcapReader(capture_file) as pcap_reader:
            for scapy_pkt in pcap_reader:
                raw_pkt = self.parse_scapy_packet(scapy_pkt)
                if raw_pkt:
                    last_time = raw_pkt.timestamp
                    terminated_flows = self.process_packet(raw_pkt)
                    flows.extend(terminated_flows)

        # Flush any remaining active flows at the end of the pcap
        remaining = self.flush_all()
        flows.extend(remaining)

        # Sort chronologically by start timestamp
        flows.sort(key=lambda x: x["timestamp"])
        return flows
