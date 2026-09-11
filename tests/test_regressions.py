"""Regression coverage for operational, temporal-state, and evidence defects."""
import asyncio
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from scapy.all import IP, TCP, PcapReader, wrpcap
from api import create_app, WebSocketManager
import json
from starlette.requests import Request
from starlette.datastructures import UploadFile
from fastapi import HTTPException
from io import BytesIO
from api import UnblockRequest
from capture.sniffer import PacketRingBuffer, PacketSniffer, RawPacket
from engine.flow_engine import FlowEngine, RawPacketInfo
from engine.streaming_runtime import StreamingModelRuntime, PersistentIPMapper
from engine.stateful_tgn import StatefulCyberTGN
from engine.hybrid_detector import HybridDetector
from engine.mitigation import MitigationController
from features.flow_aggregator import FlowRecord
from graph.graph_formatter import GraphFormatter
from graph.temporal_windows import TemporalGraphBuilder
from calibration.ece import compute_ece
from calibration.conformal import MultiClassSplitConformal
from engine.conformal import SplitConformalPredictor
from forecasting.kalman_filter import StateKalmanFilter
from forensics.pcap_dumper import ForensicPCAPDumper
from ledger.blockchain_ledger import BlockchainLedger
from pipeline import CyberDefensePipeline


def flow(t=100, src='192.0.2.1', dst='192.0.2.2'):
    return dict(src_ip=src, dst_ip=dst, timestamp=t, features=[0.] * 16)


def packet(t=100, port=80):
    return RawPacketInfo(t, '192.0.2.1', '192.0.2.2', 1234, port, 'tcp', 60, 2)


class OperationalRegressions(unittest.TestCase):
    def test_timeouts_split_reused_tuple_and_count_once(self):
        engine = FlowEngine(idle_timeout_sec=5)
        engine.process_packet(packet(100))
        engine.process_packet(packet(100, 443))
        completed = engine.process_packet(packet(106))
        self.assertEqual(len(completed), 2)
        self.assertEqual(engine.total_flows_flushed, 2)
        self.assertEqual(engine.flush_all()[0]['features'][1], 1)

    def test_pcap_honors_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.pcap'
            packets = [IP(src='192.0.2.1', dst='192.0.2.2') / TCP() for _ in range(2)]
            packets[0].time, packets[1].time = 100, 200
            wrpcap(str(path), packets)
            self.assertEqual(len(FlowEngine().process_pcap_file(str(path))), 2)

    def test_empty_shared_ring_preserved(self):
        ring = PacketRingBuffer()
        self.assertIs(PacketSniffer(ring).ring_buffer, ring)

    def test_forensic_capture_is_readable_and_keeps_time(self):
        ring = PacketRingBuffer()
        captured = IP(src='192.0.2.1', dst='192.0.2.2') / TCP(sport=1234, dport=80, flags='S')
        captured.time = 100.125
        ring.append(PacketSniffer.parse_scapy_packet(captured))
        other = captured.copy()
        other[TCP].dport = 443
        ring.append(PacketSniffer.parse_scapy_packet(other))
        alert = SimpleNamespace(alert_id='sample', timestamp=101, src_ip='192.0.2.1', dst_ip='192.0.2.2',
                                src_port=1234, dst_port=80, protocol='tcp', extra={})
        with tempfile.TemporaryDirectory() as tmp:
            dumper = ForensicPCAPDumper(Path(tmp), ring)
            path, digest = dumper.dump_alert_pcap(alert)
            self.assertEqual(digest, hashlib.sha256(path.read_bytes()).hexdigest())
            with PcapReader(str(path)) as reader:
                packets = list(reader)
            self.assertEqual(len(packets), 1)
            self.assertAlmostEqual(float(packets[0].time), 100.125)
            self.assertEqual(packets[0][TCP].dport, 80)
            ring.clear()
            self.assertEqual(dumper.dump_alert_pcap(alert), (None, ''))

    def test_firewall_refresh_does_not_duplicate_rule(self):
        controller = MitigationController(execute_system_firewall=True)
        alert = SimpleNamespace(final_score=.95, src_ip='192.0.2.1', dst_ip='192.0.2.2',
                                threat_type='TEST', explanation='test')
        with patch('engine.mitigation.subprocess.run') as run:
            controller.evaluate_and_respond(alert)
            controller.evaluate_and_respond(alert)
            self.assertEqual(run.call_count, 1)
            self.assertTrue(controller.unblock_ip(alert.src_ip))
            self.assertEqual(run.call_count, 2)
        with patch('engine.mitigation.subprocess.run', side_effect=OSError('failed')):
            incident = controller.evaluate_and_respond(alert)
            self.assertEqual(incident.action_taken, 'BLOCK_FAILED')
            self.assertFalse(controller.is_blocked(alert.src_ip))
        self.assertEqual(controller.get_recent_incidents(0), [])

    def test_snapshot_nodes_are_frozen(self):
        builder = TemporalGraphBuilder()
        builder.add_interaction('192.0.2.1', '192.0.2.2', 1234, 80, 100, np.zeros(32))
        snapshot = builder.close_snapshot()
        builder.add_interaction('192.0.2.1', '192.0.2.2', 1234, 80, 101, np.zeros(32))
        self.assertEqual(snapshot.nodes[0].out_degree, 1)

    def test_ledger_multiple_instances_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ledger.jsonl'
            first, second = BlockchainLedger(path), BlockchainLedger(path)
            leaves = ['a' * 64]
            first.commit_batch(leaves)
            leaves[0] = 'b' * 64
            self.assertTrue(first.verify_integrity())
            second.commit_batch(['c' * 64])
            self.assertEqual(len(BlockchainLedger(path).blocks), 2)
            content = path.read_text().replace('"leaf_count": 1', '"leaf_count": 2')
            path.write_text(content)
            with self.assertRaises(ValueError):
                BlockchainLedger(path)

    def test_ece_includes_probability_one(self):
        result = compute_ece(np.array([0]), np.array([1.]))
        self.assertEqual(result['ece'], 1.)
        self.assertEqual(sum(b['sample_count'] for b in result['bins']), 1)

    def test_conformal_small_samples_ties_and_uncalibrated(self):
        multi = MultiClassSplitConformal(alpha=.1)
        self.assertIsNone(multi.predict_conformal_set(np.array([.5, .5]))['coverage_guarantee'])
        multi.calibrate(np.array([0]), np.array([[1., 0.]]))
        self.assertEqual(multi.predict_conformal_set(np.array([0., 1.]))['set_size'], 2)
        binary = SplitConformalPredictor()
        self.assertFalse(binary.evaluate_flow(.99)['conformal_confirmed'])
        binary.calibrate(np.ones(100) * .9)
        self.assertFalse(binary.evaluate_flow(.9)['conformal_confirmed'])
        self.assertTrue(binary.evaluate_flow(.99)['conformal_confirmed'])
        with self.assertRaises(ValueError):
            multi.calibrate(np.array([3]), np.array([[.5, .5]]))

    def test_kalman_correction_uses_external_prediction(self):
        a, b = StateKalmanFilter(2), StateKalmanFilter(2)
        observed = np.zeros(2)
        self.assertFalse(np.allclose(a.reanchor_and_correct(np.zeros(2), observed),
                                     b.reanchor_and_correct(np.ones(2) * 100, observed)))

    def test_websocket_disconnect_during_broadcast(self):
        async def run():
            manager = WebSocketManager()
            class Socket:
                async def send_text(self, payload):
                    manager.disconnect(self)
            manager.active_connections.update([Socket(), Socket()])
            await manager.broadcast({'test': True})
            self.assertFalse(manager.active_connections)
        asyncio.run(run())


class ModelRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.streaming = StreamingModelRuntime(max_history=2)
        cls.stateful = StatefulCyberTGN(history_capacity=2)

    def test_streaming_capacity_and_validation(self):
        runtime = self.streaming
        runtime.reset_state()
        results = runtime.score_flows([flow(100 + i) for i in range(6)])
        self.assertEqual(len(results), 6)
        self.assertTrue(results[0]['cold_start'])
        self.assertTrue(results[1]['cold_start'])
        self.assertFalse(results[2]['cold_start'])
        for kwargs in [dict(batch_size=0), dict(feature_format='typo'), dict(threshold=float('nan'))]:
            with self.assertRaises(ValueError):
                runtime.score_flows([flow()], **kwargs)
        bad = flow()
        bad['features'][0] = float('nan')
        with self.assertRaises(ValueError):
            runtime.score_flows([bad])

    def test_node_capacity_resets_without_aliasing(self):
        runtime = self.streaming
        original = runtime.ip_mapper
        try:
            runtime.ip_mapper = PersistentIPMapper(2)
            runtime.reset_state()
            runtime.score_flows([flow()])
            result = runtime.score_flows([flow(101, '192.0.2.3', '192.0.2.4')])
            self.assertTrue(result[0]['cold_start'])
            self.assertEqual(set(runtime.ip_mapper.ip_to_id), {'192.0.2.3', '192.0.2.4'})
            self.assertEqual(runtime.total_flows_scored, 2)
        finally:
            runtime.ip_mapper = original
            runtime.reset_state()

    def test_stateful_history_wrap(self):
        formatter = GraphFormatter()
        for t in range(100, 108):
            record = FlowRecord('flow', '192.0.2.1', '192.0.2.2', 1234, 80, 'tcp', t, t)
            record.add_packet(RawPacket(t, record.src_ip, record.dst_ip, 1234, 80, 'tcp', 60))
            record.finalize()
            results, _ = self.stateful.process_graph_batch(formatter.format_flows([record]))
            self.assertTrue(np.isfinite(results[0]['attack_probability']))
            self.assertEqual(len(results[0]['stage_probabilities']), self.stateful.num_classes)

    def test_hybrid_reset_clears_syn_history(self):
        detector = HybridDetector()
        detector.heuristics.host_syn_history['192.0.2.1'].append(100)
        detector.reset_state()
        self.assertFalse(detector.heuristics.host_syn_history)

    def test_api_bad_capture_and_limits(self):
        async def run():
            app = create_app()
            async with app.router.lifespan_context(app):
                request = Request({'type': 'http', 'app': app})
                route = next(r for r in app.routes if r.path == '/analyze/pcap')
                with self.assertRaises(HTTPException) as error:
                    route.endpoint(request, UploadFile(filename='bad.pcap', file=BytesIO(b'junk')))
                self.assertEqual(error.exception.status_code, 400)
                with self.assertRaises(ValueError):
                    UnblockRequest(ip='invalid')
        asyncio.run(run())

class AdditionalRegressions(unittest.TestCase):
    def test_noise_filter_flags_and_empty_whitelist(self):
        from capture.filter import NoiseFilter
        packet = RawPacket(100, '192.0.2.1', '192.0.2.2', 1234, 53, 'udp', 60)
        self.assertFalse(NoiseFilter(filter_dns=False).is_noise(packet))
        self.assertEqual(NoiseFilter(whitelist_ports=set()).whitelist_ports, set())

    def test_aggregator_eviction_and_bidirectional_close(self):
        from features.flow_aggregator import FlowAggregator
        aggregate = FlowAggregator(max_active_flows=1)
        first = RawPacket(100, '192.0.2.1', '192.0.2.2', 1234, 80, 'tcp', 60)
        aggregate.add_packet(first)
        second = RawPacket(101, '192.0.2.3', '192.0.2.2', 1234, 80, 'tcp', 60, {'FIN': True})
        evicted = aggregate.add_packet(second)
        self.assertEqual(evicted.src_ip, first.src_ip)
        self.assertEqual(len(aggregate.active_flows), 1)
        reverse = RawPacket(102, second.dst_ip, second.src_ip, 80, 1234, 'tcp', 60, {'FIN': True})
        closed = aggregate.add_packet(reverse)
        self.assertEqual(len(closed.bwd_packet_lengths), 1)
        self.assertFalse(aggregate.active_flows)

    def test_pipeline_preserves_unmapped_stage_probabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = CyberDefensePipeline(forensics_dir=Path(tmp)/'pcaps', ledger_file=Path(tmp)/'ledger.jsonl')
            record = FlowRecord('f', '192.0.2.1', '192.0.2.2', 1234, 80, 'tcp', 100, 100)
            record.add_packet(RawPacket(100, record.src_ip, record.dst_ip, 1234, 80, 'tcp', 60))
            record.finalize()
            from engine.stateful_tgn import ThreatAlert
            probs = [0.] * pipeline.tgn_engine.num_classes
            probs[-1] = 1.
            alert = ThreatAlert('test', 100, record.src_ip, record.dst_ip, 1234, 80, 'tcp', .99, 1., False,
                                len(probs)-1, f'stage_{len(probs)-1}', 0., 0., record.to_16_features(), 'f',
                                extra={'stage_probabilities':probs})
            with patch.object(pipeline.tgn_engine, 'process_graph_batch', return_value=([], [alert])):
                result = pipeline.process_flows([record])
            self.assertEqual(result.alerts_generated, 1)
            self.assertFalse(result.incidents[0].mitigations_taken)
            self.assertIsNone(result.incidents[0].conformal_details['coverage_guarantee'])
