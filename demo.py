"""End-to-End Architectural Alignment Demonstration.

Demonstrates the 4 layers in action:
1. Telemetry Capture & Filtering (Captures packets, filters NTP/DNS)
2. Feature Extraction & Threat Modeling (Payload entropy, TCP SYN-ACK ratio, port roles, 16 features)
3. AI/ML Engine (Stateful CyberTGN + Conformal Bound confirmation)
4. Automated SOAR (iptables/nftables instant packet drop, quarantine, decoy rerouting)
5. Forensic Logging & Merkle Tree Commitment (PCAP dump, SHA-256 state hash, Blockchain ledger)
"""
import time
import json
import logging
from pathlib import Path
import argparse
import tempfile
from scapy.all import Ether, IP, TCP, UDP, Raw

from capture.sniffer import RawPacket
from features.flow_aggregator import FlowRecord
from pipeline import CyberDefensePipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ArchitectureDemo")


def generate_synthetic_traffic() -> list[RawPacket]:
    """Generate a realistic burst of benign traffic, noise, and high-risk attacker traffic."""
    packets = []
    base_t = time.time() - 30.0

    # 1. Benign background noise: NTP & DNS queries (Should be dropped by Layer 1 Noise Filter)
    for i in range(5):
        t = base_t + i * 0.5
        dns_pkt = RawPacket(
            timestamp=t,
            src_ip="192.168.1.105",
            dst_ip="8.8.8.8",
            src_port=54321 + i,
            dst_port=53,
            protocol="udp",
            length=64,
            payload=b"\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x07example\x03com\x00\x00\x01\x00\x01",
        )
        packets.append(dns_pkt)

    # 2. Benign web browsing (HTTP/HTTPS client <-> server)
    for i in range(12):
        t = base_t + 5.0 + i * 0.2
        is_fwd = (i % 2 == 0)
        flags = {"SYN": (i == 0), "ACK": (i > 0), "FIN": False, "RST": False, "PSH": (i > 2), "URG": False}
        payload = b"GET /index.html HTTP/1.1\r\nHost: intranet.corp\r\n\r\n" if is_fwd else b"HTTP/1.1 200 OK\r\nContent-Length: 1024\r\n"
        pkt = RawPacket(
            timestamp=t,
            src_ip="192.168.1.100" if is_fwd else "192.168.1.10",
            dst_ip="192.168.1.10" if is_fwd else "192.168.1.100",
            src_port=49812 if is_fwd else 80,
            dst_port=80 if is_fwd else 49812,
            protocol="tcp",
            length=len(payload) + 40,
            tcp_flags=flags,
            payload=payload,
            raw_frame=b"\x00" * 54 + payload,
        )
        packets.append(pkt)

    # 3. High-Risk Attacker Node (10.0.0.99) targeting Corporate SMB Server (192.168.1.10:445)
    # Malicious encrypted exploit / shellcode payload with high Shannon entropy (>7.2)
    high_entropy_shellcode = bytes([((x * 197 + 33) ^ 0xAA) % 256 for x in range(512)])
    for i in range(30):
        t = base_t + 10.0 + i * 0.05
        # High SYN count relative to ACK (SYN flood / stealth scan)
        flags = {"SYN": (i < 15), "ACK": (i >= 15), "FIN": False, "RST": False, "PSH": True, "URG": False}
        pkt = RawPacket(
            timestamp=t,
            src_ip="10.0.0.99",
            dst_ip="192.168.1.10",
            src_port=38210,
            dst_port=445,
            protocol="tcp",
            length=len(high_entropy_shellcode) + 40,
            tcp_flags=flags,
            payload=high_entropy_shellcode,
            raw_frame=b"\x00" * 54 + high_entropy_shellcode,
        )
        packets.append(pkt)

    return packets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or Path(tempfile.mkdtemp(prefix="cybertgn-demo-"))
    print(f"Demo artifacts: {output_dir}")
    print("=" * 80)
    print("   END-TO-END AUTONOMOUS CYBER DEFENSE PIPELINE DEMONSTRATION")
    print("   Architecture: Capture -> Features -> TGN + Conformal -> SOAR -> Merkle Ledger")
    print("=" * 80)

    # Initialize Pipeline with calibrated threshold of 0.85 as specified in architecture
    pipeline = CyberDefensePipeline(
        checkpoint_path=Path(__file__).resolve().parent / "checkpoints/tgn_best.pt",
        forensics_dir=output_dir / "pcaps",
        ledger_file=output_dir / "ledger.jsonl",
        dry_run_firewall=True,
        conformal_alpha=0.05,
        alert_threshold=0.85,  # High-Risk Infiltration Alert: >85% + Stage
        flow_timeout_sec=2.0,
    )

    print("\n[+] Generating synthetic network traffic (Noise + Benign Web + Attacker Node)...")
    packets = generate_synthetic_traffic()
    print(f"    Total packets captured: {len(packets)}")

    # 1. Telemetry Capture & Filtering
    print("\n[+] Step 1: Telemetry Capture & Noise Whitelist Filtering (Layer 1)")
    filtered_packets = 0
    passed_packets = 0
    all_incidents = []
    total_flows = 0
    total_alerts = 0
    last_block = None

    for pkt in packets:
        # Buffer every packet for forensic integrity
        pipeline.ring_buffer.append(pkt)
        if pipeline.noise_filter.is_noise(pkt):
            filtered_packets += 1
        else:
            passed_packets += 1
            res = pipeline.ingest_packet(pkt)
            if res:
                total_flows += res.flows_processed
                total_alerts += res.alerts_generated
                all_incidents.extend(res.incidents)
                if res.ledger_block:
                    last_block = res.ledger_block

    print(f"    - Raw packets in ring buffer:     {len(pipeline.ring_buffer)}")
    print(f"    - Noise filtered (NTP/DNS/SSDP):  {filtered_packets} packets dropped")
    print(f"    - Retained for threat modeling:   {passed_packets} packets")

    # 2. Process Remaining Flows
    print("\n[+] Step 2 & 3: Threat Modeling & Stateful CyberTGN Scoring (Layer 2 & ML Engine)")
    final_res = pipeline.flush_and_process()
    if final_res:
        total_flows += final_res.flows_processed
        total_alerts += final_res.alerts_generated
        all_incidents.extend(final_res.incidents)
        if final_res.ledger_block:
            last_block = final_res.ledger_block

    print(f"    - Total flows processed:          {total_flows}")
    print(f"    - High-risk infiltration alerts:  {total_alerts}")

    # 3. Display High-Risk Alerts & SOAR Mitigations
    if all_incidents:
        for idx, inc in enumerate(all_incidents, 1):
            print(f"\n🚨 [ALERT #{idx}] HIGH-RISK INFILTRATION DETECTED")
            print(f"   * Flow ID:             {inc.alert.flow_id}")
            print(f"   * Attacker Source:     {inc.alert.src_ip}:{inc.alert.src_port}")
            print(f"   * Victim Destination:  {inc.alert.dst_ip}:{inc.alert.dst_port} ({inc.alert.protocol.upper()})")
            print(f"   * Attack Probability:  {inc.alert.attack_probability * 100:.2f}% (Threshold: >85%)")
            print(f"   * Conformal confirmed: {inc.alert.conformal_confirmed} (requires held-out calibration)")
            print(f"   * MITRE ATT&CK Stage:  {inc.alert.stage_name} (Class {inc.alert.mitre_stage})")
            print(f"   * Payload Entropy:     {inc.alert.payload_entropy:.4f} bits/byte ({inc.alert.extra.get('entropy_class', 'HIGH')})")
            print(f"   * TCP SYN/ACK Ratio:   {inc.alert.syn_ack_ratio:.2f}")

            print(f"\n   [LAYER 3: AUTOMATED SOAR & MITIGATION]")
            for act in inc.mitigations_taken:
                status = "SUCCESS" if act.success else "FAILED"
                mode = "[DRY-RUN]" if act.is_dry_run else "[LIVE]"
                print(f"   -> Executed: {act.action_type} | Target: {act.target_ip} | Status: {status}")
                print(f"      Command:  {' '.join(act.command_executed)}")

            print(f"\n   [LAYER 4: FORENSIC LOGGING & MERKLE COMMITMENT]")
            print(f"   -> Immutable PCAP Dump: {inc.pcap_path}")
            print(f"   -> PCAP SHA-256 Hash:   {inc.pcap_hash}")
            print(f"   -> Incident State Hash: {inc.state_hash}")

    # 4. Blockchain Ledger Status
    if last_block:
        block = last_block
        print("\n" + "=" * 80)
        print("🔗 [BLOCKCHAIN LEDGER COMMITMENT]")
        print(f"   * Block Index:          #{block.block_index}")
        print(f"   * Merkle Tree Root:     {block.merkle_root}")
        print(f"   * Incident Count:       {block.leaf_count}")
        print(f"   * Previous Block Hash:  {block.prev_block_hash}")
        print(f"   * Committed Block Hash: {block.block_hash}")
        print("=" * 80)

        # Verify Blockchain Integrity
        is_valid = pipeline.ledger.verify_integrity()
        print(f"\n[✓] Cryptographic Blockchain Audit Verification: {'PASSED' if is_valid else 'FAILED'}")

    print("\n✅ Synthetic pipeline run complete. Detection quality and live enforcement remain unvalidated.")


if __name__ == "__main__":
    main()
