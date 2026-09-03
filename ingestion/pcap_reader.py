import math
from scapy.all import PcapReader, IP, TCP, UDP
from typing import Iterator
from .schema import NetworkEvent

def shannon_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log2(p_x)
    return entropy

class PcapFileLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> Iterator[NetworkEvent]:
        with PcapReader(self.filepath) as pcap_reader:
            for packet in pcap_reader:
                if IP in packet:
                    ip_layer = packet[IP]
                    src_port = 0
                    dst_port = 0
                    tcp_flags = None
                    tcp_window = None
                    protocol = 'other'
                    
                    if TCP in packet:
                        protocol = 'tcp'
                        src_port = packet[TCP].sport
                        dst_port = packet[TCP].dport
                        tcp_flags = int(packet[TCP].flags)
                        tcp_window = packet[TCP].window
                    elif UDP in packet:
                        protocol = 'udp'
                        src_port = packet[UDP].sport
                        dst_port = packet[UDP].dport
                        
                    payload = bytes(packet.payload)
                    entropy = shannon_entropy(payload)
                    
                    yield NetworkEvent(
                        timestamp=float(packet.time),
                        src_ip=ip_layer.src,
                        dst_ip=ip_layer.dst,
                        src_port=src_port,
                        dst_port=dst_port,
                        protocol=protocol,
                        packet_size=len(packet),
                        tcp_flags=tcp_flags,
                        ttl=ip_layer.ttl,
                        payload_entropy=entropy,
                        tcp_window=tcp_window
                    )
