from dataclasses import dataclass
from typing import Optional

@dataclass
class NetworkEvent:
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    protocol: str
    packet_size: int
    tcp_flags: Optional[int] = None
    ttl: Optional[int] = None
    payload_entropy: Optional[float] = None
    tcp_window: Optional[int] = None
    label: Optional[str] = None
    direction: str = 'fwd'
