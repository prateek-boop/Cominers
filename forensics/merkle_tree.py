"""Cryptographic Merkle Tree Implementation.

Provides binary Merkle tree construction over incident state hashes,
root hash calculation, and inclusion proof generation/verification.
"""
import hashlib
from typing import List, Tuple, Optional


class MerkleTree:
    """Binary SHA-256 Merkle Tree."""

    def __init__(self, leaf_hashes: List[str]):
        if not leaf_hashes:
            raise ValueError("Cannot build Merkle Tree from empty leaves list")

        self.leaf_hashes = list(leaf_hashes)
        self.levels: List[List[str]] = [self.leaf_hashes]
        self._build_tree()

    def _hash_pair(self, left: str, right: str) -> str:
        combined = left.encode("utf-8") + right.encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def _build_tree(self):
        current_level = self.leaf_hashes
        while len(current_level) > 1:
            next_level = []
            for i in range(0, len(current_level), 2):
                left = current_level[i]
                right = current_level[i + 1] if i + 1 < len(current_level) else left
                parent_hash = self._hash_pair(left, right)
                next_level.append(parent_hash)
            self.levels.append(next_level)
            current_level = next_level

    @property
    def root(self) -> str:
        """Return the hex Merkle Root."""
        return self.levels[-1][0]

    def get_proof(self, index: int) -> List[Tuple[str, str]]:
        """
        Generate inclusion proof for the leaf at `index`.
        Returns a list of (sibling_hash, position) where position is 'left' or 'right'.
        """
        if index < 0 or index >= len(self.leaf_hashes):
            raise IndexError("Leaf index out of bounds")

        proof: List[Tuple[str, str]] = []
        current_idx = index

        for level in self.levels[:-1]:
            is_right_child = (current_idx % 2 == 1)
            sibling_idx = current_idx - 1 if is_right_child else current_idx + 1

            if sibling_idx < len(level):
                sibling_hash = level[sibling_idx]
            else:
                sibling_hash = level[current_idx]  # odd-element duplicated

            position = "left" if is_right_child else "right"
            proof.append((sibling_hash, position))
            current_idx //= 2

        return proof

    @staticmethod
    def verify_proof(leaf_hash: str, proof: List[Tuple[str, str]], expected_root: str) -> bool:
        """Verify that a given leaf belongs to the Merkle tree with the expected root."""
        current_hash = leaf_hash
        for sibling_hash, position in proof:
            if position not in ("left", "right"):
                return False
            if position == "left":
                combined = sibling_hash.encode("utf-8") + current_hash.encode("utf-8")
            else:
                combined = current_hash.encode("utf-8") + sibling_hash.encode("utf-8")
            current_hash = hashlib.sha256(combined).hexdigest()

        return current_hash == expected_root
