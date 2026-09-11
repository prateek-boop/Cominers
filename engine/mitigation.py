"""Automated Incident Response & Mitigation Controller.

Enforces network defense policies, generates firewall mitigation rules,
and manages the active threat register and IP blocklist.
"""
from __future__ import annotations
import logging
import os
import subprocess
import time
import threading
from ipaddress import ip_address
from engine.flow_engine import synchronized
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class BlockedHost:
    ip: str
    threat_type: str
    score: float
    blocked_at: float
    expires_at: float
    reason: str
    active: bool = True


@dataclass
class IncidentRecord:
    incident_id: str
    timestamp: float
    attacker_ip: str
    target_ip: str
    threat_type: str
    score: float
    action_taken: str  # 'BLOCK_IP', 'ALERT', 'ALLOW'
    reason: str


class MitigationController:
    """Automated policy engine and firewall integration."""

    def __init__(self, block_threshold: float = 0.85, alert_threshold: float = 0.50,
                 default_block_ttl_sec: float = 300.0, execute_system_firewall: bool = False):
        self.lock = threading.RLock()
        self.block_threshold = block_threshold
        self.alert_threshold = alert_threshold
        self.default_block_ttl_sec = default_block_ttl_sec
        self.execute_system_firewall = execute_system_firewall

        self.blocked_ips: Dict[str, BlockedHost] = {}
        self.incident_history: List[IncidentRecord] = []
        self._incident_counter = 0

    @synchronized
    def evaluate_and_respond(self, alert_obj) -> Optional[IncidentRecord]:
        """Evaluates an alert object and executes policy action."""
        score = alert_obj.final_score
        src_ip = alert_obj.src_ip
        dst_ip = alert_obj.dst_ip
        threat_type = alert_obj.threat_type
        reason = alert_obj.explanation

        current_time = time.time()
        self._prune_expired_blocks(current_time)

        if score >= self.block_threshold:
            action = "BLOCK_IP"
            if not self._apply_block(src_ip, threat_type, score, current_time, reason):
                action = "BLOCK_FAILED"
        elif score >= self.alert_threshold:
            action = "ALERT"
        else:
            action = "ALLOW"

        if action != "ALLOW":
            self._incident_counter += 1
            incident = IncidentRecord(
                incident_id=f"INC-{int(current_time)}-{self._incident_counter:04d}",
                timestamp=current_time,
                attacker_ip=src_ip,
                target_ip=dst_ip,
                threat_type=threat_type,
                score=score,
                action_taken=action,
                reason=reason
            )
            self.incident_history.append(incident)
            # Keep history to last 500 incidents
            if len(self.incident_history) > 500:
                self.incident_history = self.incident_history[-500:]
            return incident

        return None

    def _apply_block(self, ip: str, threat_type: str, score: float, current_time: float, reason: str):
        ip = str(ip_address(ip))
        if ip not in self.blocked_ips and self.execute_system_firewall:
            if not self._exec_iptables_drop(ip):
                return False
        expires = current_time + self.default_block_ttl_sec
        self.blocked_ips[ip] = BlockedHost(
            ip=ip,
            threat_type=threat_type,
            score=score,
            blocked_at=current_time,
            expires_at=expires,
            reason=reason,
            active=True
        )
        logger.warning(f"[MITIGATION] IP {ip} BLOCKED for {self.default_block_ttl_sec}s. Threat: {threat_type} (Score {score:.2f})")

        return True

    def _firewall_rule(self, ip: str, operation: str) -> bool:
        address = ip_address(ip)
        binary = "ip6tables" if address.version == 6 else "iptables"
        try:
            privilege_prefix = [] if os.geteuid() == 0 else ["sudo", "-n"]
            subprocess.run(privilege_prefix + [binary, operation, "INPUT", "-s", str(address), "-j", "DROP"],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            return True
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("Firewall operation %s failed for %s: %s", operation, ip, exc)
            return False

    def _exec_iptables_drop(self, ip: str):
        return self._firewall_rule(ip, "-I")

    @synchronized
    def unblock_ip(self, ip: str) -> bool:
        """Remove the register entry only after the system rule is removed."""
        ip = str(ip_address(ip))
        if ip not in self.blocked_ips:
            return False
        if self.execute_system_firewall and not self._firewall_rule(ip, "-D"):
            return False
        del self.blocked_ips[ip]
        return True

    def _prune_expired_blocks(self, current_time: float):
        expired = [ip for ip, b in self.blocked_ips.items() if current_time >= b.expires_at]
        for ip in expired:
            self.unblock_ip(ip)

    @synchronized
    def is_blocked(self, ip: str) -> bool:
        self._prune_expired_blocks(time.time())
        return ip in self.blocked_ips

    @synchronized
    def get_active_blocks(self) -> List[Dict[str, Any]]:
        self._prune_expired_blocks(time.time())
        return [asdict(b) for b in self.blocked_ips.values()]

    @synchronized
    def get_recent_incidents(self, limit: int = 50) -> List[Dict[str, Any]]:
        if limit <= 0:
            return []
        return [asdict(inc) for inc in reversed(self.incident_history[-limit:])]
