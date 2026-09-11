"""Layer 3: Automated SOAR & Mitigation Layer.

Provides real-time packet dropping, host isolation, and honeypot/decoy rerouting
via iptables, nftables, and eBPF/XDP drop hooks.
Includes safe dry-run mode and mitigation action auditing.
"""
import logging
import subprocess
import shutil
from dataclasses import dataclass
import time
from ipaddress import ip_address, ip_network
from typing import List, Optional

logger = logging.getLogger("SOAR.Mitigation")


@dataclass
class MitigationAction:
    action_type: str  # "PACKET_DROP", "HOST_ISOLATE", "DECOY_REROUTE"
    target_ip: str
    command_executed: List[str]
    timestamp: float
    success: bool
    details: str
    is_dry_run: bool


class FirewallMitigator:
    """Automated firewall and network controller for incident response."""

    def __init__(self, dry_run: bool = True, preferred_backend: str = "auto"):
        """
        Args:
            dry_run: If True, logs commands without modifying system networking.
            preferred_backend: "iptables", "nft", "ebpf", or "auto".
        """
        self.dry_run = dry_run
        self.history: List[MitigationAction] = []

        if preferred_backend == "auto":
            if shutil.which("nft"):
                self.backend = "nft"
            elif shutil.which("iptables"):
                self.backend = "iptables"
            else:
                self.backend = "simulated"
        else:
            self.backend = preferred_backend

        logger.info(f"Initialized FirewallMitigator with backend={self.backend}, dry_run={self.dry_run}")

    def _execute(self, cmd: List[str], action_type: str, target_ip: str, details: str) -> MitigationAction:
        target_ip = str(ip_address(target_ip))
        ts = time.time()
        if self.dry_run:
            logger.info(f"[DRY-RUN SOAR] Would execute: {' '.join(cmd)}")
            action = MitigationAction(
                action_type=action_type,
                target_ip=target_ip,
                command_executed=cmd,
                timestamp=ts,
                success=True,
                details=f"[DRY RUN] {details}",
                is_dry_run=True,
            )
            self.history.append(action)
            return action

        try:
            logger.warning(f"[LIVE SOAR EXECUTION] Executing: {' '.join(cmd)}")
            res = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=10)
            action = MitigationAction(
                action_type=action_type,
                target_ip=target_ip,
                command_executed=cmd,
                timestamp=ts,
                success=True,
                details=f"Success: {res.stdout.strip()}",
                is_dry_run=False,
            )
        except (subprocess.SubprocessError, OSError) as e:
            err_msg = str(e)
            if hasattr(e, "stderr") and e.stderr:
                err_msg += f" - {e.stderr}"
            logger.error(f"Mitigation failed for {target_ip}: {err_msg}")
            action = MitigationAction(
                action_type=action_type,
                target_ip=target_ip,
                command_executed=cmd,
                timestamp=ts,
                success=False,
                details=f"Error: {err_msg}",
                is_dry_run=False,
            )

        self.history.append(action)
        return action

    def instant_packet_drop(self, attacker_ip: str) -> MitigationAction:
        """Drop all incoming packets from the attacker node immediately."""
        if self.backend == "nft":
            cmd = ["nft", "add", "rule", "inet", "filter", "input", "ip", "saddr", attacker_ip, "drop"]
        elif self.backend == "iptables":
            cmd = ["iptables", "-I", "INPUT", "-s", attacker_ip, "-j", "DROP"]
        else:
            # eBPF / simulated hook
            cmd = ["ebpf-xdp-ctl", "drop-ip", attacker_ip]

        return self._execute(cmd, "PACKET_DROP", attacker_ip, f"Instant packet drop for malicious IP {attacker_ip}")

    def isolate_host_quarantine(self, victim_ip: str, mgmt_subnet: str = "192.168.1.0/24") -> MitigationAction:
        """Quarantine a compromised host by cutting outbound connections except management."""
        if self.backend == "nft":
            cmd = [
                "nft", "add", "rule", "inet", "filter", "forward",
                "ip", "saddr", victim_ip, "ip", "daddr", "!=", mgmt_subnet, "drop"
            ]
        elif self.backend == "iptables":
            cmd = [
                "iptables", "-I", "FORWARD", "-s", victim_ip, "!", "-d", mgmt_subnet, "-j", "DROP"
            ]
        else:
            cmd = ["vlan-ctl", "quarantine-vlan", victim_ip]

        return self._execute(
            cmd, "HOST_ISOLATE", victim_ip, f"Host quarantine applied to {victim_ip} preserving {mgmt_subnet}"
        )

    def reroute_to_decoy(self, attacker_ip: str, target_port: int, decoy_port: int = 8088) -> MitigationAction:
        """Reroute incoming malicious requests from attacker into an isolated decoy/honeypot listener."""
        if self.backend == "iptables":
            cmd = [
                "iptables", "-t", "nat", "-I", "PREROUTING", "-p", "tcp",
                "-s", attacker_ip, "--dport", str(target_port),
                "-j", "REDIRECT", "--to-ports", str(decoy_port)
            ]
        elif self.backend == "nft":
            cmd = [
                "nft", "add", "rule", "inet", "nat", "prerouting",
                "ip", "saddr", attacker_ip, "tcp", "dport", str(target_port),
                "redirect", "to", str(decoy_port)
            ]
        else:
            cmd = ["honeypot-rerouter", "--attacker", attacker_ip, "--port", str(decoy_port)]

        return self._execute(
            cmd, "DECOY_REROUTE", attacker_ip,
            f"Rerouting attacker {attacker_ip}:{target_port} -> Decoy/Honeypot port {decoy_port}"
        )
