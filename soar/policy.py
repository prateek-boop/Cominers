"""SOAR Policy Engine.

Evaluates high-risk threat alerts and automatically triggers calibrated mitigation responses:
- Reconnaissance -> Decoy / Honeypot rerouting or logging
- Infiltration / Lateral Movement / Impact -> Instant packet drop & Host quarantine
"""
import logging
from typing import List, Optional
from engine.stateful_tgn import ThreatAlert
from .mitigation import FirewallMitigator, MitigationAction

logger = logging.getLogger("SOAR.Policy")


class SOARPolicyEngine:
    """Orchestrates automated mitigation policies based on AI/ML threat alerts."""

    def __init__(
        self,
        mitigator: Optional[FirewallMitigator] = None,
        enable_instant_drop: bool = True,
        enable_host_quarantine: bool = True,
        enable_decoy_rerouting: bool = True,
        decoy_port: int = 8088,
    ):
        self.mitigator = mitigator or FirewallMitigator(dry_run=True)
        self.enable_instant_drop = enable_instant_drop
        self.enable_host_quarantine = enable_host_quarantine
        self.enable_decoy_rerouting = enable_decoy_rerouting
        self.decoy_port = decoy_port

    def handle_alert(self, alert: ThreatAlert) -> List[MitigationAction]:
        """Evaluate alert against SOAR policy and execute mitigations."""
        actions: List[MitigationAction] = []

        stage_info = alert.extra.get('stage_conformal', {})
        stages = set(stage_info.get('conformal_set', []))
        critical = {'lateral_movement', 'command_and_control'}
        allowed = (
            alert.conformal_confirmed and alert.attack_probability > 0.85
            and alert.extra.get('model_attack_probability', 0.0) > 0.85
            and alert.extra.get('stage_validation_passed') is True
            and stage_info.get('is_calibrated') is True
            and bool(stages) and stages <= critical and alert.stage_name in stages
        )
        if not allowed:
            logger.info(f"Alert {alert.alert_id} did not meet high-risk threshold for automated SOAR.")
            return actions

        logger.warning(
            f"[SOAR ACTIVATION] High-Risk Alert {alert.alert_id} | Attacker: {alert.src_ip} -> "
            f"Victim: {alert.dst_ip} | Stage: {alert.stage_name} | Prob: {alert.attack_probability:.4f}"
        )

        stage = alert.stage_name.lower()

        # 1. Reconnaissance Policy -> Honeypot / Decoy Rerouting
        if stage == "reconnaissance" and self.enable_decoy_rerouting:
            action = self.mitigator.reroute_to_decoy(
                attacker_ip=alert.src_ip,
                target_port=alert.dst_port,
                decoy_port=self.decoy_port,
            )
            actions.append(action)

        # 2. Infiltration / Lateral Movement / C2 / Impact Policy -> Instant Drop & Quarantine
        elif stage in (
            "initial_access",
            "credential_access",
            "lateral_movement",
            "command_and_control",
            "exfiltration",
            "impact",
        ) or stage.startswith("stage_") or stage.startswith("unmapped_"):
            # Instant packet drop on attacker node
            if self.enable_instant_drop:
                drop_action = self.mitigator.instant_packet_drop(alert.src_ip)
                actions.append(drop_action)

            # Quarantine target victim if lateral movement or impact detected
            if self.enable_host_quarantine and stages == {'lateral_movement'}:
                quarantine_action = self.mitigator.isolate_host_quarantine(alert.dst_ip)
                actions.append(quarantine_action)

        return actions
