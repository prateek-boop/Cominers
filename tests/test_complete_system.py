"""Comprehensive test suite for the complete CyberTGN NIDS/NIPS platform."""
from __future__ import annotations
import asyncio
import json
import os
import tempfile
import time
import unittest

from scapy.all import Ether, IP, TCP, UDP, wrpcap

from api import create_app
from engine.flow_engine import FlowEngine, RawPacketInfo
from engine.hybrid_detector import HybridDetector, FastHeuristicEngine
from engine.mitigation import MitigationController
from engine.streaming_runtime import StreamingModelRuntime


class CompleteSystemTests(unittest.TestCase):

    def test_flow_engine_handshake_and_features(self):
        """Verify 3-way handshake tracking and 16 CICFlowMeter feature calculations."""
        engine = FlowEngine()
        t0 = 1700000000.0

        # SYN
        p1 = RawPacketInfo(t0, '192.168.1.100', '10.0.0.5', 40000, 80, 'tcp', 60, tcp_flags=0x02)
        # SYN-ACK
        p2 = RawPacketInfo(t0 + 0.01, '10.0.0.5', '192.168.1.100', 80, 40000, 'tcp', 60, tcp_flags=0x12)
        # ACK
        p3 = RawPacketInfo(t0 + 0.02, '192.168.1.100', '10.0.0.5', 40000, 80, 'tcp', 54, tcp_flags=0x10)
        # Data packet
        p4 = RawPacketInfo(t0 + 0.05, '192.168.1.100', '10.0.0.5', 40000, 80, 'tcp', 400, tcp_flags=0x18)

        for p in [p1, p2, p3, p4]:
            engine.process_packet(p)

        flows = engine.flush_all()
        self.assertEqual(len(flows), 1)
        flow = flows[0]

        # 16 features must be present and finite
        self.assertEqual(len(flow['features']), 16)
        for val in flow['features']:
            self.assertTrue(isinstance(val, (int, float)))
            self.assertFalse(val != val)  # not NaN

        # TCP handshake confirmed
        self.assertTrue(flow['metadata']['handshake_complete'])
        self.assertEqual(flow['metadata']['fwd_packets'], 3)
        self.assertEqual(flow['metadata']['bwd_packets'], 1)

    def test_fast_heuristics_syn_flood(self):
        """Verify immediate heuristic detection of a TCP SYN flood."""
        heuristics = FastHeuristicEngine()
        # Create a SYN flood flow signature: 20 fwd packets, 0 bwd packets, tiny packet size
        syn_flood_flow = {
            "src_ip": "203.0.113.5",
            "dst_ip": "10.0.0.1",
            "timestamp": time.time(),
            "metadata": {"protocol": "tcp", "dst_port": 80},
            "features": [
                10000.0,  # Duration (10ms)
                20.0,     # Fwd packets
                0.0,      # Bwd packets
                1200.0,   # Tot Fwd bytes
                0.0,      # Tot Bwd bytes
                60.0,     # Fwd length mean
                0.0,      # Bwd length mean
                120000.0, # Bytes/s
                2000.0,   # Packets/s
                500.0,    # Flow IAT
                500.0,    # Fwd IAT
                0.0,      # Bwd IAT
                2000.0,   # Fwd Packets/s
                0.0,      # Bwd Packets/s
                60.0,     # Avg packet size (<= 70)
                0.0       # Down/Up ratio (0)
            ]
        }
        verdict = heuristics.evaluate_flow(syn_flood_flow)
        self.assertTrue(verdict.is_threat)
        self.assertEqual(verdict.threat_type, "TCP_SYN_FLOOD")
        self.assertGreater(verdict.confidence, 0.85)

    def test_fast_heuristics_port_scan(self):
        """Verify port scan heuristic when scanning multiple ports."""
        heuristics = FastHeuristicEngine(scan_port_threshold=5)
        src = "198.51.100.99"
        dst = "10.0.0.2"
        now = time.time()

        verdict = None
        for port in range(1, 8):
            flow = {
                "src_ip": src,
                "dst_ip": dst,
                "timestamp": now + (port * 0.1),
                "metadata": {"protocol": "tcp", "dst_port": port},
                "features": [1000.0, 1.0, 0.0, 60.0, 0.0, 60.0, 0.0, 60000.0, 1000.0, 0.0, 0.0, 0.0, 1000.0, 0.0, 60.0, 0.0]
            }
            verdict = heuristics.evaluate_flow(flow)

        self.assertIsNotNone(verdict)
        self.assertTrue(verdict.is_threat)
        self.assertEqual(verdict.threat_type, "PORT_SCAN")

    def test_mitigation_controller_blocks_attacker(self):
        """Verify policy engine automatically blocks high-confidence threats."""
        controller = MitigationController(block_threshold=0.85, default_block_ttl_sec=60.0)

        class MockAlert:
            final_score = 0.95
            src_ip = "198.51.100.77"
            dst_ip = "10.0.0.1"
            threat_type = "TCP_SYN_FLOOD"
            explanation = "Unidirectional TCP SYN flood"

        incident = controller.evaluate_and_respond(MockAlert())
        self.assertIsNotNone(incident)
        self.assertEqual(incident.action_taken, "BLOCK_IP")
        self.assertTrue(controller.is_blocked("198.51.100.77"))

        # Manual unblock
        unblocked = controller.unblock_ip("198.51.100.77")
        self.assertTrue(unblocked)
        self.assertFalse(controller.is_blocked("198.51.100.77"))

    def test_pcap_processing_and_hybrid_detection(self):
        """Generates a real .pcap file with Scapy, processes with FlowEngine and HybridDetector."""
        with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as tmp:
            pcap_path = tmp.name

        try:
            packets = []
            base_time = 1700000000.0
            # Generate 15 SYN packets (SYN flood probe)
            for i in range(15):
                pkt = (
                    Ether() /
                    IP(src="192.168.1.50", dst="10.0.0.80") /
                    TCP(sport=50000 + i, dport=80, flags="S", seq=1000 + i)
                )
                pkt.time = base_time + (i * 0.01)
                packets.append(pkt)

            wrpcap(pcap_path, packets)

            # 1. Process with FlowEngine
            engine = FlowEngine()
            flows = engine.process_pcap_file(pcap_path)
            self.assertGreater(len(flows), 0)

            # 2. Run Hybrid Detector
            detector = HybridDetector()
            alerts = detector.analyze_flows(flows)
            self.assertEqual(len(alerts), len(flows))

            # Validate that SYN flood probe was caught
            attacks = [a for a in alerts if a.is_attack]
            self.assertGreater(len(attacks), 0)
            threat = attacks[0]
            self.assertTrue(threat.is_attack)
            self.assertIn("SYN", threat.threat_type)

            # 3. Validate Mitigation Controller
            controller = MitigationController()
            incident = controller.evaluate_and_respond(threat)
            self.assertIsNotNone(incident)
            self.assertTrue(controller.is_blocked(threat.src_ip))

        finally:
            if os.path.exists(pcap_path):
                os.remove(pcap_path)

    def test_api_endpoints_asgi(self):
        """Tests new API routes directly using ASGI interface."""
        async def run_asgi():
            app = create_app()
            async with app.router.lifespan_context(app):
                async def request(method, path, body=None, headers=None):
                    data = json.dumps(body).encode() if body is not None and isinstance(body, dict) else (body or b'')
                    req_headers = [(b'content-type', b'application/json')] if body is not None else []
                    if headers:
                        req_headers.extend(headers)
                    sent = []

                    async def receive():
                        return {'type': 'http.request', 'body': data, 'more_body': False}

                    async def send(message):
                        sent.append(message)

                    scope = {
                        'type': 'http', 'asgi': {'version': '3.0'}, 'http_version': '1.1',
                        'method': method, 'scheme': 'http', 'path': path, 'raw_path': path.encode(),
                        'query_string': b'', 'root_path': '', 'headers': req_headers,
                        'client': ('127.0.0.1', 123), 'server': ('test', 80)
                    }
                    await app(scope, receive, send)
                    status = next(m['status'] for m in sent if m['type'] == 'http.response.start')
                    content = b''.join(m.get('body', b'') for m in sent if m['type'] == 'http.response.body')
                    return status, content

                # 1. Test Dashboard HTML route
                status, body = await request('GET', '/dashboard')
                self.assertEqual(status, 200)
                self.assertIn(b'CyberTGN Operations Center', body)

                # 2. Test Sniffer Status route
                status, body = await request('GET', '/sniff/status')
                self.assertEqual(status, 200)
                info = json.loads(body)
                self.assertIn('is_running', info)
                self.assertFalse(info['is_running'])

                # 3. Test Alerts route
                status, body = await request('GET', '/alerts')
                self.assertEqual(status, 200)
                alerts_info = json.loads(body)
                self.assertIn('blocked_ips', alerts_info)
                self.assertIn('recent_incidents', alerts_info)

                # 4. Test Model Reset route
                status, body = await request('POST', '/model/reset', body={})
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(body)['status'], 'reset_complete')

        asyncio.run(run_asgi())


if __name__ == '__main__':
    unittest.main()
