"""Actual packet capture and firewall verification in a disposable network namespace.

Run: python3 scripts/run_isolated.py
Never run against the host namespace. This runner fails before any mutation if
it cannot verify namespace isolation.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def command(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True, timeout=15).stdout


def main():
    host_namespace = os.environ.get('CYBERTGN_HOST_NET_NS')
    if not host_namespace or os.readlink('/proc/self/ns/net') == host_namespace:
        raise RuntimeError('Refusing to change the host network namespace')
    command('ip', 'link', 'set', 'lo', 'up')
    from engine.mitigation import MitigationController
    from engine.hybrid_detector import HybridDetector
    from engine.live_sniffer import LiveTrafficSniffer
    from capture.sniffer import PacketRingBuffer, PacketSniffer
    from soar.mitigation import FirewallMitigator

    results = {}
    def check(name, test):
        try:
            details = test()
            results[name] = {'passed': True, 'details': details}
        except Exception as exc:
            results[name] = {'passed': False, 'error': f'{type(exc).__name__}: {exc}'}
        print(json.dumps({name: results[name]}), flush=True)

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 61000))
    receiver.settimeout(.2)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender.bind(('127.0.0.2', 0))

    def delivered():
        sender.sendto(b'namespace-only-probe', receiver.getsockname())
        try:
            receiver.recvfrom(2048)
            return True
        except socket.timeout:
            return False

    def controller_test():
        controller = MitigationController(execute_system_firewall=True, default_block_ttl_sec=.3)
        alert = SimpleNamespace(final_score=.99, src_ip='127.0.0.2', dst_ip='127.0.0.1',
                                threat_type='ISOLATED_TEST', explanation='Isolated UDP test')
        assert delivered(), 'Baseline traffic must arrive'
        incident = controller.evaluate_and_respond(alert)
        assert incident.action_taken == 'BLOCK_IP', incident
        assert not delivered(), 'Blocked packet reached the listener'
        controller.evaluate_and_respond(alert)
        rules = command('iptables', '-S', 'INPUT')
        assert rules.count('-s 127.0.0.2/32') == 1, rules
        time.sleep(.35)
        assert not controller.is_blocked(alert.src_ip), 'Expired rule remains registered'
        assert delivered(), 'Packet still blocked after TTL cleanup'
        controller.evaluate_and_respond(alert)
        assert controller.unblock_ip(alert.src_ip)
        assert delivered(), 'Packet still blocked after manual unblock'
        return 'Real UDP delivery -> drop -> TTL delivery -> manual unblock delivery'

    check('controller_drop_ttl_unblock', controller_test)
    command('iptables', '-F', 'INPUT')

    def soar_iptables():
        assert delivered()
        action = FirewallMitigator(dry_run=False, preferred_backend='iptables').instant_packet_drop('127.0.0.2')
        assert action.success, action.details
        assert not delivered()
        command('iptables', '-D', 'INPUT', '-s', '127.0.0.2', '-j', 'DROP')
        assert delivered()
        return 'Real iptables rule blocked packets'
    check('soar_iptables_drop', soar_iptables)
    command('iptables', '-F', 'INPUT')

    def soar_nft():
        command('nft', 'add', 'table', 'inet', 'filter')
        command('nft', 'add', 'chain', 'inet', 'filter', 'input', '{ type filter hook input priority 0; policy accept; }')
        assert delivered()
        action = FirewallMitigator(dry_run=False, preferred_backend='nft').instant_packet_drop('127.0.0.2')
        assert action.success, action.details
        assert not delivered()
        command('nft', 'delete', 'table', 'inet', 'filter')
        assert delivered()
        return 'Real nftables rule blocked packets'
    check('soar_nft_drop', soar_nft)
    subprocess.run(['nft', 'delete', 'table', 'inet', 'filter'], capture_output=True)

    def capture_test():
        detector = HybridDetector()
        sniffer = LiveTrafficSniffer(detector, MitigationController(), interface='lo', bpf_filter='udp port 61000')
        received = []
        sniffer.on_flow_scored = received.append
        try:
            sniffer.start()
            deadline = time.monotonic() + 5
            while sniffer.packets_captured == 0 and time.monotonic() < deadline:
                assert delivered()
                time.sleep(.1)
            assert sniffer.packets_captured > 0, sniffer.get_status()
            for _ in range(25):
                assert delivered()
                time.sleep(.002)
        finally:
            sniffer.stop()
        assert sniffer._thread is None or not sniffer._thread.is_alive(), 'Worker did not stop'
        assert not sniffer.last_error, sniffer.last_error
        assert received, 'Captured packets never reached model scoring'
        assert any(a.is_attack for a in received), 'UDP burst did not raise an alert'
        status = sniffer.get_status()
        assert status['active_flows_in_memory'] == 0, status
        return status
    check('live_capture_score_dispatch_shutdown', capture_test)

    def raw_capture():
        ring = PacketRingBuffer()
        sniffer = PacketSniffer(ring)
        sniffer.start_live_capture('lo', bpf_filter='udp port 61000')
        try:
            deadline = time.monotonic() + 5
            while not len(ring) and time.monotonic() < deadline:
                assert delivered()
                time.sleep(.1)
            assert len(ring), 'No packets in shared ring'
        finally:
            sniffer.stop_live_capture()
        assert not sniffer._capture.running
        return {'packets': len(ring), 'stopped_without_extra_packet': True}
    check('raw_capture_shared_buffer_shutdown', raw_capture)
    def forwarded_paths():
        # Two disposable network namespaces connected through this test router.
        children = [subprocess.Popen(['unshare', '--net', 'sleep', '120']) for _ in range(2)]
        echo = None
        details = []
        try:
            time.sleep(.15)
            for child in children:
                assert child.poll() is None
                assert os.readlink(f'/proc/{child.pid}/ns/net') != os.readlink('/proc/self/ns/net')
            client, server = children
            for idx, (child, prefix) in enumerate(zip(children, ('192.0.2', '198.51.100'))):
                router_if, peer_if = f'router{idx}', f'peer{idx}'
                command('ip', 'link', 'add', router_if, 'type', 'veth', 'peer', 'name', peer_if)
                command('ip', 'link', 'set', peer_if, 'netns', str(child.pid))
                command('ip', 'addr', 'add', prefix + '.1/24', 'dev', router_if)
                command('ip', 'link', 'set', router_if, 'up')
                command('nsenter', '-t', str(child.pid), '-n', 'ip', 'link', 'set', 'lo', 'up')
                command('nsenter', '-t', str(child.pid), '-n', 'ip', 'addr', 'add', prefix + '.2/24', 'dev', peer_if)
                command('nsenter', '-t', str(child.pid), '-n', 'ip', 'link', 'set', peer_if, 'up')
                command('nsenter', '-t', str(child.pid), '-n', 'ip', 'route', 'add', 'default', 'via', prefix + '.1')
            command('sysctl', '-w', 'net.ipv4.ip_forward=1')
            server_code = 'import socket\ns=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)\ns.bind(("198.51.100.2",61010))\nwhile True:\n data,addr=s.recvfrom(1024)\n s.sendto(data,addr)\n'
            echo = subprocess.Popen(['nsenter', '-t', str(server.pid), '-n', sys.executable, '-c', server_code])
            time.sleep(.2)
            probe_code = "import socket; s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.settimeout(.4); s.sendto(b'probe',('198.51.100.2',61010)); assert s.recv(1024)==b'probe'"
            def probe():
                return subprocess.run(['nsenter', '-t', str(client.pid), '-n', sys.executable, '-c', probe_code], capture_output=True, timeout=3).returncode == 0
            assert probe(), 'Forwarding baseline failed'
            for backend in ('iptables', 'nft'):
                if backend == 'nft':
                    command('nft', 'add', 'table', 'inet', 'filter')
                    command('nft', 'add', 'chain', 'inet', 'filter', 'forward', '{ type filter hook forward priority 0; policy accept; }')
                action = FirewallMitigator(dry_run=False, preferred_backend=backend).isolate_host_quarantine('192.0.2.2')
                assert action.success, action.details
                assert not probe(), backend + ' quarantine did not block forwarded traffic'
                if backend == 'nft':
                    command('nft', 'delete', 'table', 'inet', 'filter')
                else:
                    command('iptables', '-F', 'FORWARD')
                assert probe(), backend + ' traffic did not recover'
                details.append(backend + ' quarantine blocks forwarded packets')

            # Test REDIRECT through PREROUTING using a client outside this router.
            for backend in ('iptables', 'nft'):
                listener = socket.socket()
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                listener.bind(('0.0.0.0', 61012))
                listener.listen(1)
                listener.settimeout(2)
                if backend == 'nft':
                    command('nft', 'add', 'table', 'inet', 'nat')
                    command('nft', 'add', 'chain', 'inet', 'nat', 'prerouting', '{ type nat hook prerouting priority -100; policy accept; }')
                action = FirewallMitigator(dry_run=False, preferred_backend=backend).reroute_to_decoy('192.0.2.2', 61011, 61012)
                assert action.success, action.details
                tcp_probe = "import socket; s=socket.create_connection(('192.0.2.1',61011),timeout=1); s.sendall(b'decoy'); s.close()"
                command('nsenter', '-t', str(client.pid), '-n', sys.executable, '-c', tcp_probe)
                connection, _ = listener.accept()
                with connection:
                    assert connection.recv(100) == b'decoy'
                listener.close()
                if backend == 'nft':
                    command('nft', 'delete', 'table', 'inet', 'nat')
                else:
                    command('iptables', '-t', 'nat', '-F', 'PREROUTING')
                details.append(backend + ' redirect delivers to decoy port')
            return details
        finally:
            if echo:
                echo.terminate()
                echo.wait(timeout=5)
            for child in children:
                child.terminate()
                child.wait(timeout=5)
    check('forwarding_quarantine_and_decoy', forwarded_paths)

    receiver.close()
    sender.close()
    output = ROOT / 'reports/audit/live-verification.json'
    output.write_text(json.dumps(results, indent=2) + '\n')
    return int(any(not r['passed'] for r in results.values()))

if __name__ == '__main__':
    sys.exit(main())
