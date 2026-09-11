"""Unit and Integration Tests for the End-to-End Cyber Defense Architecture.

Verifies:
- Layer 1: PacketRingBuffer, RawPacket parsing, NoiseFilter (NTP/DNS whitelist)
- Layer 2: Shannon Payload Entropy, TCP Handshake Metrics, Port Roles, Flow Aggregation (16 features)
- AI/ML Engine: SplitConformalPredictor, StatefulCyberTGN memory & scoring
- Layer 3: FirewallMitigator (instant drop, quarantine, decoy), SOARPolicyEngine
- Layer 4: ForensicPCAPDumper, SHA-256 state hashing, MerkleTree proofs, BlockchainLedger integrity
- End-to-End Pipeline integration
"""
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

from capture.sniffer import RawPacket, PacketRingBuffer, PacketSniffer
from capture.filter import NoiseFilter
from features.entropy import shannon_entropy, classify_entropy
from features.handshake import HandshakeMetrics
from features.port_roles import classify_port_roles
from features.flow_aggregator import FlowAggregator, FlowRecord
from graph.graph_formatter import GraphFormatter
from engine.conformal import SplitConformalPredictor
from engine.stateful_tgn import StatefulCyberTGN, ThreatAlert
from soar.mitigation import FirewallMitigator, MitigationAction
from soar.policy import SOARPolicyEngine
from forensics.pcap_dumper import ForensicPCAPDumper
from forensics.hasher import compute_alert_state_hash
from forensics.merkle_tree import MerkleTree
from ledger.blockchain_ledger import BlockchainLedger
from pipeline import CyberDefensePipeline


class TestLayer1Telemetry(unittest.TestCase):
    def test_noise_filter_whitelisting(self):
        filter_eng = NoiseFilter()
        # DNS should be filtered
        dns_pkt = RawPacket(
            timestamp=100.0, src_ip="10.0.0.1", dst_ip="8.8.8.8",
            src_port=52341, dst_port=53, protocol="udp", length=60
        )
        self.assertTrue(filter_eng.is_noise(dns_pkt))

        # NTP should be filtered
        ntp_pkt = RawPacket(
            timestamp=100.0, src_ip="10.0.0.1", dst_ip="129.6.15.28",
            src_port=123, dst_port=123, protocol="udp", length=76
        )
        self.assertTrue(filter_eng.is_noise(ntp_pkt))

        # Attacker HTTP/SMB probe should NOT be filtered
        attack_pkt = RawPacket(
            timestamp=100.0, src_ip="10.0.0.99", dst_ip="192.168.1.10",
            src_port=44512, dst_port=445, protocol="tcp", length=120
        )
        self.assertFalse(filter_eng.is_noise(attack_pkt))

    def test_ring_buffer_retention(self):
        ring = PacketRingBuffer(capacity=5)
        for i in range(10):
            pkt = RawPacket(
                timestamp=float(i), src_ip=f"10.0.0.{i}", dst_ip="192.168.1.1",
                src_port=1000 + i, dst_port=80, protocol="tcp", length=64
            )
            ring.append(pkt)

        self.assertEqual(len(ring), 5)
        # Verify oldest packets were evicted and newest retained
        matched = ring.get_packets_for_flow("10.0.0.9", "192.168.1.1")
        self.assertEqual(len(matched), 1)


class TestLayer2FeatureExtraction(unittest.TestCase):
    def test_shannon_entropy(self):
        empty_data = b""
        self.assertEqual(shannon_entropy(empty_data), 0.0)

        # Repetitive data -> low entropy
        low_data = b"AAAAAAAAAA"
        self.assertAlmostEqual(shannon_entropy(low_data), 0.0)

        # Uniform byte distribution -> maximum entropy (~8.0)
        high_data = bytes(range(256))
        self.assertAlmostEqual(shannon_entropy(high_data), 8.0, places=2)
        self.assertEqual(classify_entropy(8.0), "HIGH_ENCRYPTED_OR_PACKED")

    def test_handshake_metrics(self):
        tracker = HandshakeMetrics()
        # 3 SYNs without ACK (SYN flood / scan)
        for i in range(3):
            tracker.update_packet(is_forward=True, tcp_flags={"SYN": True, "ACK": False}, timestamp=10.0 + i)
        self.assertEqual(tracker.syn_count, 3)
        self.assertEqual(tracker.ack_count, 0)
        self.assertGreater(tracker.syn_ack_ratio, 2.5)

        # Complete handshake
        tracker.update_packet(is_forward=False, tcp_flags={"SYN": True, "ACK": True}, timestamp=13.0)
        tracker.update_packet(is_forward=True, tcp_flags={"SYN": False, "ACK": True}, timestamp=13.1)
        self.assertTrue(tracker.handshake_completed)
        self.assertIsNotNone(tracker.handshake_latency)

    def test_port_roles(self):
        profile = classify_port_roles(src_port=52341, dst_port=443)
        self.assertEqual(profile.src_tier, "DYNAMIC_EPHEMERAL")
        self.assertEqual(profile.dst_tier, "SYSTEM_WELL_KNOWN")
        self.assertEqual(profile.service_name, "HTTPS")
        self.assertTrue(profile.is_client_to_server)

    def test_flow_aggregator_16_features(self):
        agg = FlowAggregator()
        for i in range(4):
            pkt = RawPacket(
                timestamp=100.0 + i * 0.1,
                src_ip="192.168.1.5",
                dst_ip="10.0.0.1",
                src_port=50000,
                dst_port=80,
                protocol="tcp",
                length=100,
                tcp_flags={"SYN": (i == 0), "ACK": (i > 0)},
                payload=b"test",
            )
            agg.add_packet(pkt)

        flows = agg.flush_all()
        self.assertEqual(len(flows), 1)
        features = flows[0].to_16_features()
        self.assertEqual(len(features), 16)
        self.assertTrue(all(np.isfinite(f) for f in features))


class TestAIMLEngine(unittest.TestCase):
    def test_conformal_predictor(self):
        conformal = SplitConformalPredictor(alpha=0.05)
        # Calibrate on benign predictions
        benign_scores = np.linspace(0.01, 0.40, 100)
        conformal.calibrate(benign_scores)
        self.assertTrue(conformal.is_calibrated)
        self.assertLess(conformal.q_hat, 0.45)

        # High risk attack flow (0.95) should be confirmed
        eval_attack = conformal.evaluate_flow(0.95, base_threshold=0.85)
        self.assertTrue(eval_attack["conformal_confirmed"])
        self.assertTrue(eval_attack["is_high_risk_alert"])
        self.assertLess(eval_attack["conformal_p_value"], 0.05)

        # Benign flow (0.10) should not trigger
        eval_benign = conformal.evaluate_flow(0.10, base_threshold=0.85)
        self.assertFalse(eval_benign["is_high_risk_alert"])

    def test_stateful_tgn_initialization(self):
        engine = StatefulCyberTGN(alert_threshold=0.5)
        self.assertEqual(engine.msg_dim, 16)
        self.assertGreater(engine.num_nodes, 0)


class TestLayer3SOAR(unittest.TestCase):
    def test_firewall_mitigator_dry_run(self):
        mitigator = FirewallMitigator(dry_run=True, preferred_backend="iptables")
        action = mitigator.instant_packet_drop("198.51.100.99")
        self.assertTrue(action.success)
        self.assertTrue(action.is_dry_run)
        self.assertEqual(action.action_type, "PACKET_DROP")
        self.assertIn("198.51.100.99", action.command_executed)

        iso_action = mitigator.isolate_host_quarantine("192.168.1.50")
        self.assertTrue(iso_action.success)
        self.assertEqual(iso_action.action_type, "HOST_ISOLATE")

        decoy_action = mitigator.reroute_to_decoy("198.51.100.99", 80, 8088)
        self.assertTrue(decoy_action.success)
        self.assertEqual(decoy_action.action_type, "DECOY_REROUTE")


class TestLayer4ForensicsAndLedger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.forensics_dir = Path(self.temp_dir) / "dumps"
        self.ledger_file = Path(self.temp_dir) / "ledger.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_merkle_tree_proofs(self):
        leaves = [f"leaf_hash_{i}" for i in range(8)]
        tree = MerkleTree(leaves)
        self.assertIsNotNone(tree.root)
        self.assertEqual(len(tree.root), 64)

        # Verify inclusion proof for leaf 3
        proof = tree.get_proof(3)
        self.assertTrue(MerkleTree.verify_proof(leaves[3], proof, tree.root))

    def test_blockchain_ledger_integrity(self):
        ledger = BlockchainLedger(ledger_file=self.ledger_file)
        block1 = ledger.commit_batch(["hash_a", "hash_b"])
        self.assertEqual(block1.block_index, 0)
        self.assertEqual(block1.prev_block_hash, "0" * 64)

        block2 = ledger.commit_batch(["hash_c", "hash_d"])
        self.assertEqual(block2.block_index, 1)
        self.assertEqual(block2.prev_block_hash, block1.block_hash)

        self.assertTrue(ledger.verify_integrity())


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.forensics_dir = Path(self.temp_dir) / "dumps"
        self.ledger_file = Path(self.temp_dir) / "ledger.jsonl"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_end_to_end_execution(self):
        pipeline = CyberDefensePipeline(
            dry_run_firewall=True,
            forensics_dir=self.forensics_dir,
            ledger_file=self.ledger_file,
            alert_threshold=0.0,  # Ensure alert triggers for testing
        )
        # Calibrate conformal predictor so test packet triggers high-risk alert
        pipeline.tgn_engine.conformal.calibrate(np.zeros(100))

        pkt = RawPacket(
            timestamp=100.0,
            src_ip="198.51.100.44",
            dst_ip="192.168.1.10",
            src_port=48123,
            dst_port=445,
            protocol="tcp",
            length=128,
            tcp_flags={"SYN": True, "ACK": False, "FIN": True, "RST": True},
            payload=b"\x90" * 64,
            raw_frame=b"\x00" * 54 + b"\x90" * 64,
        )
        pipeline.ring_buffer.append(pkt)
        cycle = pipeline.ingest_packet(pkt)
        if cycle is None:
            cycle = pipeline.flush_and_process()

        self.assertIsNotNone(cycle)
        self.assertGreaterEqual(cycle.flows_processed, 1)
        self.assertGreaterEqual(cycle.alerts_generated, 1)
        self.assertIsNotNone(cycle.ledger_block)
        self.assertTrue(pipeline.ledger.verify_integrity())


if __name__ == "__main__":
    unittest.main()
