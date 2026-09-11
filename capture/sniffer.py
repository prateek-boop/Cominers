"""Layer 1: Telemetry Capture Layer.

Provides promiscuous sniffing (Scapy / raw socket), PCAP reading,
and a circular packet ring buffer for immutable forensic PCAP dumping.
"""
from dataclasses import dataclass, field
from collections import deque
import threading
import time
from typing import Iterator, Optional, Callable
from scapy.all import IP, TCP, UDP, ICMP, Raw, PcapReader, AsyncSniffer


@dataclass
class RawPacket:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    length: int
    tcp_flags: dict = field(default_factory=dict)
    payload: bytes = b""
    raw_frame: bytes = b""
    linktype: int = 1


class PacketRingBuffer:
    """Thread-safe circular ring buffer for storing raw packets for forensic PCAP retrieval."""

    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.Lock()

    def append(self, packet: RawPacket):
        with self.lock:
            self.buffer.append(packet)

    def get_packets_for_flow(
        self,
        src_ip: str,
        dst_ip: str,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        src_port: Optional[int] = None,
        dst_port: Optional[int] = None,
        protocol: Optional[str] = None,
    ) -> list[RawPacket]:
        """Extract all raw packets matching the bidirectional conversation."""
        matched = []
        with self.lock:
            for pkt in self.buffer:
                if start_time is not None and pkt.timestamp < start_time:
                    continue
                if end_time is not None and pkt.timestamp > end_time:
                    continue
                if (pkt.src_ip == src_ip and pkt.dst_ip == dst_ip) or (
                    pkt.src_ip == dst_ip and pkt.dst_ip == src_ip
                ):
                    if protocol is not None and pkt.protocol != protocol:
                        continue
                    if src_port is not None and dst_port is not None:
                        forward = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port) == (src_ip, src_port, dst_ip, dst_port)
                        reverse = (pkt.src_ip, pkt.src_port, pkt.dst_ip, pkt.dst_port) == (dst_ip, dst_port, src_ip, src_port)
                        if not (forward or reverse):
                            continue
                    matched.append(pkt)
        return matched

    def clear(self):
        with self.lock:
            self.buffer.clear()

    def __len__(self):
        with self.lock:
            return len(self.buffer)


class PacketSniffer:
    """Captures packets from a network interface or parses them from a PCAP file."""

    def __init__(self, ring_buffer: Optional[PacketRingBuffer] = None):
        self.ring_buffer = ring_buffer if ring_buffer is not None else PacketRingBuffer()
        self._running = False
        self._sniffer_thread = None
        self._capture = None

    @staticmethod
    def parse_scapy_packet(pkt) -> Optional[RawPacket]:
        """Convert a Scapy packet into a structured RawPacket."""
        if IP not in pkt:
            return None

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        length = len(pkt)
        timestamp = float(pkt.time) if hasattr(pkt, "time") else time.time()
        # Normalize captured evidence to raw IP, independent of source link type.
        raw_frame = bytes(ip_layer)

        protocol = "other"
        src_port = 0
        dst_port = 0
        tcp_flags = {
            "SYN": False,
            "ACK": False,
            "FIN": False,
            "RST": False,
            "PSH": False,
            "URG": False,
        }
        payload = b""

        if TCP in pkt:
            protocol = "tcp"
            tcp_layer = pkt[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            flags = int(tcp_layer.flags)
            tcp_flags = {
                "FIN": bool(flags & 0x01),
                "SYN": bool(flags & 0x02),
                "RST": bool(flags & 0x04),
                "PSH": bool(flags & 0x08),
                "ACK": bool(flags & 0x10),
                "URG": bool(flags & 0x20),
            }
            if Raw in pkt:
                payload = bytes(pkt[Raw].load)
        elif UDP in pkt:
            protocol = "udp"
            udp_layer = pkt[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)
            if Raw in pkt:
                payload = bytes(pkt[Raw].load)
        elif ICMP in pkt:
            protocol = "icmp"

        return RawPacket(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            src_port=src_port,
            dst_port=dst_port,
            protocol=protocol,
            length=length,
            tcp_flags=tcp_flags,
            payload=payload,
            raw_frame=raw_frame,
            linktype=101,
        )

    def read_pcap(self, pcap_path: str) -> Iterator[RawPacket]:
        """Iterate through packets in a PCAP file and record to ring buffer."""
        with open(pcap_path, "rb") as capture_file, PcapReader(capture_file) as reader:
            for pkt in reader:
                raw_pkt = self.parse_scapy_packet(pkt)
                if raw_pkt:
                    self.ring_buffer.append(raw_pkt)
                    yield raw_pkt

    def start_live_capture(
        self,
        interface: str = "any",
        callback: Optional[Callable[[RawPacket], None]] = None,
        packet_count: int = 0,
        bpf_filter: str = "",
    ):
        """Start live promiscuous sniffing on the specified interface."""
        if self._capture and self._capture.running:
            return
        self._running = True

        def _handler(pkt):
            if not self._running:
                return
            raw_pkt = self.parse_scapy_packet(pkt)
            if raw_pkt:
                self.ring_buffer.append(raw_pkt)
                if callback:
                    callback(raw_pkt)

        self._capture = AsyncSniffer(
            iface=interface if interface != "any" else None,
            prn=_handler, count=packet_count, filter=bpf_filter, store=False,
        )
        self._capture.start()

    def stop_live_capture(self):
        self._running = False
        if self._capture and self._capture.running:
            self._capture.stop()
