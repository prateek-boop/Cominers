"""TCP Handshake Metrics Tracker.

Computes SYN-ACK ratios, handshake completion status, and connection anomalies.
Disproportionate SYN-to-ACK ratios (>3.0) signal SYN flood attacks, port scans,
or half-open stealth scans.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class HandshakeMetrics:
    syn_count: int = 0
    syn_ack_count: int = 0
    ack_count: int = 0
    rst_count: int = 0
    fin_count: int = 0
    handshake_completed: bool = False
    syn_ack_ratio: float = 0.0
    first_syn_time: Optional[float] = None
    first_ack_time: Optional[float] = None
    handshake_latency: Optional[float] = None

    def update_packet(self, is_forward: bool, tcp_flags: dict, timestamp: float):
        syn = tcp_flags.get("SYN", False)
        ack = tcp_flags.get("ACK", False)
        rst = tcp_flags.get("RST", False)
        fin = tcp_flags.get("FIN", False)

        if syn and not ack and is_forward:
            self.syn_count += 1
            if self.first_syn_time is None:
                self.first_syn_time = timestamp
        elif syn and ack and not is_forward:
            self.syn_ack_count += 1
        elif ack and not syn:
            self.ack_count += 1
            if is_forward and self.syn_count > 0 and self.syn_ack_count > 0 and self.first_ack_time is None:
                self.first_ack_time = timestamp
                self.handshake_completed = True
                if self.first_syn_time is not None:
                    self.handshake_latency = max(0.0, self.first_ack_time - self.first_syn_time)

        if rst:
            self.rst_count += 1
        if fin:
            self.fin_count += 1

        # Calculate SYN-ACK ratio: SYN / (ACK + 1) to prevent division by zero
        self.syn_ack_ratio = float(self.syn_count / (self.ack_count + 1))
