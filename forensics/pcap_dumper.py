"""Forensic PCAP Dumper.

Extracts matching raw packet frames from the circular ring buffer for any flagged high-risk flow
and dumps an immutable local PCAP file for forensic analysis and cryptographic commitment.
"""
from pathlib import Path
import hashlib
import logging
from typing import Optional, Tuple
from scapy.all import wrpcap, Ether, IP, TCP, UDP, Raw

from capture.sniffer import PacketRingBuffer, RawPacket
from engine.stateful_tgn import ThreatAlert

logger = logging.getLogger("Forensics.PCAPDumper")


class ForensicPCAPDumper:
    """Dumps raw packets into immutable PCAP files per incident."""

    def __init__(self, output_dir: Path = Path("forensics/dumps"), ring_buffer: Optional[PacketRingBuffer] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ring_buffer = ring_buffer

    def dump_alert_pcap(
        self,
        alert: ThreatAlert,
        time_padding_sec: float = 5.0,
    ) -> Tuple[Optional[Path], str]:
        """
        Extract packets from ring buffer matching the alert and dump to PCAP.
        Returns (pcap_file_path, pcap_sha256_hash).
        """
        pcap_path = self.output_dir / f"alert_{alert.alert_id}_{int(alert.timestamp)}.pcap"

        if self.ring_buffer is None:
            return None, ""

        start_t = alert.extra.get("flow_start_time", alert.timestamp) - time_padding_sec
        raw_packets = self.ring_buffer.get_packets_for_flow(
            src_ip=alert.src_ip, dst_ip=alert.dst_ip,
            src_port=alert.src_port, dst_port=alert.dst_port, protocol=alert.protocol,
            start_time=start_t, end_time=alert.timestamp + time_padding_sec,
        )
        scapy_pkts = []
        for rp in raw_packets:
            if not rp.raw_frame:
                continue
            try:
                # Write a consistent raw-IP PCAP without fabricating transport data.
                pkt = IP(rp.raw_frame) if rp.linktype == 101 else Ether(rp.raw_frame)[IP].copy()
                pkt.time = rp.timestamp
                scapy_pkts.append(pkt)
            except (ValueError, IndexError):
                logger.warning("Skipping undecodable evidence frame")
        if not scapy_pkts:
            return None, ""
        wrpcap(str(pcap_path), scapy_pkts)
        return pcap_path, hashlib.sha256(pcap_path.read_bytes()).hexdigest()
