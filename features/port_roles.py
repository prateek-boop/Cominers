"""Port Role Classification.

Categorizes ports into standardized architectural tiers:
- SYSTEM_WELL_KNOWN: 0 - 1023 (e.g., HTTP 80, HTTPS 443, SSH 22, SMB 445)
- REGISTERED_SERVICE: 1024 - 49151 (e.g., MySQL 3306, RDP 3389, Web Proxy 8080)
- DYNAMIC_EPHEMERAL: 49152 - 65535 (Outbound client source ports, peer-to-peer)
"""
from dataclasses import dataclass
from typing import Tuple


SYSTEM_SERVICES = {
    21: "FTP",
    22: "SSH",
    23: "Telnet",
    25: "SMTP",
    53: "DNS",
    80: "HTTP",
    110: "POP3",
    143: "IMAP",
    443: "HTTPS",
    445: "SMB",
    3389: "RDP",
    3306: "MySQL",
    5432: "PostgreSQL",
    8080: "HTTP-Alt",
    8443: "HTTPS-Alt",
}


@dataclass
class PortProfile:
    src_port: int
    dst_port: int
    src_tier: str
    dst_tier: str
    service_name: str
    is_client_to_server: bool
    is_suspicious_ephemeral_pair: bool


def get_port_tier(port: int) -> str:
    if 0 <= port <= 1023:
        return "SYSTEM_WELL_KNOWN"
    elif 1024 <= port <= 49151:
        return "REGISTERED_SERVICE"
    else:
        return "DYNAMIC_EPHEMERAL"


def classify_port_roles(src_port: int, dst_port: int) -> PortProfile:
    src_tier = get_port_tier(src_port)
    dst_tier = get_port_tier(dst_port)

    service_name = SYSTEM_SERVICES.get(dst_port) or SYSTEM_SERVICES.get(src_port) or "UNKNOWN"

    # Typical pattern: client dynamic port (>=1024) -> server system/registered port (<=1024 or registered)
    is_client_to_server = (dst_port in SYSTEM_SERVICES) or (dst_port < src_port)

    # Ephemeral to ephemeral with no known service could indicate stealth C2 or P2P
    is_suspicious_ephemeral_pair = (
        src_tier == "DYNAMIC_EPHEMERAL"
        and dst_tier == "DYNAMIC_EPHEMERAL"
        and service_name == "UNKNOWN"
    )

    return PortProfile(
        src_port=src_port,
        dst_port=dst_port,
        src_tier=src_tier,
        dst_tier=dst_tier,
        service_name=service_name,
        is_client_to_server=is_client_to_server,
        is_suspicious_ephemeral_pair=is_suspicious_ephemeral_pair,
    )
