"""Cryptographic State Hasher.

Computes canonical SHA-256 state hashes combining threat telemetry,
AI/ML decision metrics, and immutable forensic PCAP hashes.
"""
import hashlib
import json
from engine.stateful_tgn import ThreatAlert


def compute_alert_state_hash(alert: ThreatAlert, pcap_hash: str) -> str:
    """
    Produce a deterministic cryptographic SHA-256 state hash for an incident.
    Hash = SHA256(alert_id || timestamp || src_ip || dst_ip || attack_prob || stage || pcap_hash || features)
    """
    state_payload = {
        "alert_id": alert.alert_id,
        "timestamp": float(alert.timestamp),
        "src_ip": alert.src_ip,
        "dst_ip": alert.dst_ip,
        "src_port": alert.src_port,
        "dst_port": alert.dst_port,
        "protocol": alert.protocol,
        "attack_probability": float(alert.attack_probability),
        "conformal_p_value": float(alert.conformal_p_value),
        "payload_entropy": float(alert.payload_entropy),
        "syn_ack_ratio": float(alert.syn_ack_ratio),
        "conformal_confirmed": alert.conformal_confirmed,
        "mitre_stage": alert.mitre_stage,
        "stage_name": alert.stage_name,
        "pcap_hash": pcap_hash,
        "features": [float(f) for f in alert.features_16],
    }

    canonical_json = json.dumps(state_payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
