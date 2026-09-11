"""Regression tests found by actual networking and ledger verification."""
import fcntl
import hashlib
import json
import multiprocessing
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from ledger.blockchain_ledger import BlockchainLedger
from engine.mitigation import MitigationController


def write_blocks(path, worker):
    ledger = BlockchainLedger(Path(path))
    for index in range(10):
        ledger.commit_batch([hashlib.sha256(f'{worker}:{index}'.encode()).hexdigest()])


class OperationalVerification(unittest.TestCase):
    def test_privileged_controller_does_not_require_sudo(self):
        controller = MitigationController(execute_system_firewall=True)
        with patch('engine.mitigation.os.geteuid', return_value=0), patch('engine.mitigation.subprocess.run') as run:
            self.assertTrue(controller._exec_iptables_drop('192.0.2.1'))
            self.assertEqual(run.call_args.args[0][0], 'iptables')
        with patch('engine.mitigation.os.geteuid', return_value=1000), patch('engine.mitigation.subprocess.run') as run:
            self.assertTrue(controller._exec_iptables_drop('192.0.2.1'))
            self.assertEqual(run.call_args.args[0][:2], ['sudo', '-n'])

    def test_ledger_four_concurrent_writers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'ledger.jsonl')
            context = multiprocessing.get_context('fork')
            workers = [context.Process(target=write_blocks, args=(path, i)) for i in range(4)]
            for worker in workers:
                worker.start()
            for worker in workers:
                worker.join(timeout=15)
                self.assertEqual(worker.exitcode, 0)
            ledger = BlockchainLedger(Path(path))
            self.assertEqual(len(ledger.blocks), 40)
            self.assertEqual(len({b.leaf_hashes[0] for b in ledger.blocks}), 40)
            self.assertTrue(ledger.verify_integrity())

    def test_reader_waits_for_in_progress_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'ledger.jsonl'
            ledger = BlockchainLedger(path)
            block = ledger.commit_batch(['a'*64])
            serialized = json.dumps(block.__dict__) + '\n'
            results, failures = [], []
            def read():
                try:
                    results.append(BlockchainLedger(path).verify_integrity())
                except Exception as exc:
                    failures.append(str(exc))
            with open(str(path)+'.lock','a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                path.write_text(serialized[:20])
                reader = threading.Thread(target=read)
                reader.start()
                time.sleep(.05)
                self.assertTrue(reader.is_alive())
                path.write_text(serialized)
                fcntl.flock(lock, fcntl.LOCK_UN)
            reader.join(timeout=3)
            self.assertFalse(failures)
            self.assertEqual(results, [True])
