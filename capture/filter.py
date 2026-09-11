"""Layer 1 Noise Filter: Protocol & Service Whitelisting.

Filters high-volume, non-threat network background traffic (NTP, DNS, mDNS, SSDP, DHCP)
to prevent downstream feature pollution and false positives.
"""
from typing import Set, Tuple
from .sniffer import RawPacket


class NoiseFilter:
    """Configurable whitelist filter for network telemetry."""

    DEFAULT_WHITELIST_PORTS: Set[int] = {
        53,    # DNS
        5353,  # mDNS
        123,   # NTP
        67,    # DHCP server
        68,    # DHCP client
        1900,  # SSDP
        137,   # NetBIOS Name Service
        138,   # NetBIOS Datagram
    }

    DEFAULT_WHITELIST_IPS: Set[str] = {
        "255.255.255.255",
        "224.0.0.251",  # mDNS multicast
        "239.255.255.250", # SSDP multicast
    }

    def __init__(
        self,
        whitelist_ports: Set[int] = None,
        whitelist_ips: Set[str] = None,
        whitelist_ip_pairs: Set[Tuple[str, str]] = None,
        filter_dns: bool = True,
        filter_ntp: bool = True,
        filter_broadcast: bool = True,
    ):
        self.whitelist_ports = set(self.DEFAULT_WHITELIST_PORTS if whitelist_ports is None else whitelist_ports)
        self.whitelist_ips = set(self.DEFAULT_WHITELIST_IPS if whitelist_ips is None else whitelist_ips)
        self.whitelist_ip_pairs = set(whitelist_ip_pairs or set())
        self.filter_dns = filter_dns
        self.filter_ntp = filter_ntp
        self.filter_broadcast = filter_broadcast

    def is_noise(self, pkt: RawPacket) -> bool:
        """Return True if packet matches benign noise signatures."""
        # Check IP whitelist (broadcast / multicast addresses)
        if self.filter_broadcast:
            if pkt.dst_ip in self.whitelist_ips or pkt.src_ip in self.whitelist_ips:
                return True
            if pkt.dst_ip.startswith("224.") or pkt.dst_ip.startswith("239."):
                return True

        # Check explicit IP pair whitelist
        if (pkt.src_ip, pkt.dst_ip) in self.whitelist_ip_pairs or (
            pkt.dst_ip, pkt.src_ip
        ) in self.whitelist_ip_pairs:
            return True

        # Check DNS
        if self.filter_dns and (pkt.src_port in (53, 5353) or pkt.dst_port in (53, 5353)):
            return True

        # Check NTP
        if self.filter_ntp and (pkt.src_port == 123 or pkt.dst_port == 123):
            return True

        if not self.filter_dns and (pkt.src_port in (53, 5353) or pkt.dst_port in (53, 5353)):
            return False
        if not self.filter_ntp and (pkt.src_port == 123 or pkt.dst_port == 123):
            return False

        # Check other whitelisted ports
        if pkt.src_port in self.whitelist_ports or pkt.dst_port in self.whitelist_ports:
            return True

        return False

    def filter_packet(self, pkt: RawPacket) -> bool:
        """Convenience method: returns True if packet should be retained (not noise)."""
        return not self.is_noise(pkt)
