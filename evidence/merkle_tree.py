import hashlib
from typing import List

class MerkleTree:
    def __init__(self, leaves: List[bytes]):
        self.leaves = leaves
        self.tree = self._build_tree(leaves)
        
    def _build_tree(self, nodes: List[bytes]) -> List[List[bytes]]:
        if not nodes:
            return []
        tree = [nodes]
        while len(tree[-1]) > 1:
            level = tree[-1]
            next_level = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i+1] if i+1 < len(level) else left
                combined = left + right
                next_level.append(hashlib.sha256(combined).digest())
            tree.append(next_level)
        return tree
        
    @property
    def root(self) -> str:
        return self.tree[-1][0].hex() if self.tree else ""
