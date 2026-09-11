"""Blockchain Ledger for Immutable Merkle State Commitments.

Provides an append-only cryptographic chain of blocks committing Merkle roots
of threat alerts and forensic PCAP state hashes.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import time
import threading
import fcntl
import os
from typing import List, Optional

from forensics.merkle_tree import MerkleTree


@dataclass
class LedgerBlock:
    block_index: int
    timestamp: float
    prev_block_hash: str
    merkle_root: str
    leaf_count: int
    leaf_hashes: List[str]
    block_hash: str

    def calculate_hash(self) -> str:
        header = {
            "block_index": self.block_index,
            "timestamp": self.timestamp,
            "prev_block_hash": self.prev_block_hash,
            "merkle_root": self.merkle_root,
            "leaf_count": self.leaf_count,
        }
        header_str = json.dumps(header, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(header_str.encode("utf-8")).hexdigest()


class BlockchainLedger:
    """Manages an append-only JSON Lines cryptographic ledger of Merkle state blocks."""

    GENESIS_PREV_HASH = "0" * 64

    def __init__(self, ledger_file: Path = Path("ledger/ledger.jsonl")):
        self.ledger_file = Path(ledger_file)
        self.ledger_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.blocks: List[LedgerBlock] = []
        # Readers must not observe an in-progress append from another process.
        with open(str(self.ledger_file) + ".lock", "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_SH)
            try:
                self._load_ledger()
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _load_ledger(self):
        self.blocks.clear()
        if not self.ledger_file.exists():
            return

        with open(self.ledger_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                block = LedgerBlock(**data)
                self.blocks.append(block)
        if not self.verify_integrity():
            raise ValueError("Ledger integrity verification failed")

    def commit_batch(self, leaf_hashes: List[str]) -> LedgerBlock:
        with self.lock, open(str(self.ledger_file) + ".lock", "a") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                self._load_ledger()
                return self._commit_batch(list(leaf_hashes))
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def _commit_batch(self, leaf_hashes: List[str]) -> LedgerBlock:
        """
        Commit a batch of incident state hashes into a new blockchain block.
        Calculates Merkle Root and appends block to ledger.
        """
        if not leaf_hashes:
            raise ValueError("Cannot commit empty leaf batch to ledger")

        merkle_tree = MerkleTree(leaf_hashes)
        merkle_root = merkle_tree.root

        block_index = len(self.blocks)
        prev_hash = self.blocks[-1].block_hash if self.blocks else self.GENESIS_PREV_HASH
        timestamp = time.time()

        block = LedgerBlock(
            block_index=block_index,
            timestamp=timestamp,
            prev_block_hash=prev_hash,
            merkle_root=merkle_root,
            leaf_count=len(leaf_hashes),
            leaf_hashes=leaf_hashes,
            block_hash="",
        )
        block.block_hash = block.calculate_hash()

        # Append to persistent ledger
        with open(self.ledger_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(block)) + "\n")
            f.flush()
            os.fsync(f.fileno())

        self.blocks.append(block)
        return block

    def verify_integrity(self) -> bool:
        """Verify the cryptographic chain of all blocks and Merkle roots."""
        if not self.blocks:
            return True

        for i, block in enumerate(self.blocks):
            if block.block_index != i or block.leaf_count != len(block.leaf_hashes) or not block.leaf_hashes:
                return False
            # Check block hash validity
            expected_hash = block.calculate_hash()
            if block.block_hash != expected_hash:
                return False

            # Check previous hash pointer
            if i == 0:
                if block.prev_block_hash != self.GENESIS_PREV_HASH:
                    return False
            else:
                if block.prev_block_hash != self.blocks[i - 1].block_hash:
                    return False

            # Check Merkle root
            tree = MerkleTree(block.leaf_hashes)
            if tree.root != block.merkle_root:
                return False

        return True
