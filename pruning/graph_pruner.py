"""Phase 1: Adaptive Graph Pruner.

Drops high-frequency, deterministic benign background traffic (NTP, local ARP, safe internal DNS)
before graph construction layer, preventing graph memory explosion.
"""
from typing import Iterator, Optional, Any, Dict
from capture.sniffer import RawPacket


class AdaptiveGraphPruner:
    """Filters routine benign background protocols before temporal graph construction."""

    def __init__(self, drop_ntp: bool = True, drop_arp: bool = True, drop_internal_dns: bool = True):
        self.drop_ntp = drop_ntp
        self.drop_arp = drop_arp
        self.drop_internal_dns = drop_internal_dns
        self.dropped_packets_count = 0
        self.retained_packets_count = 0

    def should_prune(self, pkt: RawPacket) -> bool:
        """Evaluates whether a packet should be dropped as background noise."""
        proto = (pkt.protocol or "").lower()
        sport = pkt.src_port or 0
        dport = pkt.dst_port or 0

        # NTP (UDP 123)
        if self.drop_ntp and proto == "udp" and (sport == 123 or dport == 123):
            return True

        # ARP / Local broadcast
        if self.drop_arp and proto in ("arp", "rarp"):
            return True

        # Safe internal DNS queries (UDP 53) - unless suspected DNS tunneling
        # Small DNS queries with standard payload size (< 120 bytes) are routine
        if self.drop_internal_dns and proto == "udp" and (sport == 53 or dport == 53):
            if pkt.length < 120:
                return True

        # Multicast / Discovery protocols
        if dport in (5353, 1900, 5355, 137, 138):  # mDNS, SSDP, LLMNR, NetBIOS
            return True

        return False

    def filter_packet(self, pkt: RawPacket) -> Optional[RawPacket]:
        if self.should_prune(pkt):
            self.dropped_packets_count += 1
            return None
        self.retained_packets_count += 1
        return pkt

    def filter_stream(self, packets: Iterator[RawPacket]) -> Iterator[RawPacket]:
        for pkt in packets:
            filtered = self.filter_packet(pkt)
            if filtered is not None:
                yield filtered

    # Alias for compatibility
    prune_stream = filter_stream

    def get_prune_stats(self) -> dict:
        total = self.dropped_packets_count + self.retained_packets_count
        ratio = (self.dropped_packets_count / total) if total > 0 else 0.0
        return {
            "total_examined": total,
            "total_pruned": self.dropped_packets_count,
            "retained": self.retained_packets_count,
            "prune_ratio": ratio,
        }

