"""Shannon Payload Entropy Calculation.

Quantifies randomness of packet and flow payloads.
High entropy (>7.0) typically indicates encrypted C2, packed binaries, or encrypted exfiltration.
Low entropy (<2.0) typically indicates plain text commands, zero-padded buffers, or repetitive beacons.
"""
import math
from typing import Union


def shannon_entropy(data: Union[bytes, bytearray]) -> float:
    """Calculate Shannon entropy in bits per byte (range: [0.0, 8.0])."""
    if not data:
        return 0.0

    length = len(data)
    # Count frequency of each byte (0-255)
    counts = [0] * 256
    for b in data:
        counts[b] += 1

    entropy = 0.0
    for count in counts:
        if count > 0:
            p = count / length
            entropy -= p * math.log2(p)

    return float(entropy)


def classify_entropy(entropy: float) -> str:
    """Classify payload entropy level."""
    if entropy >= 7.2:
        return "HIGH_ENCRYPTED_OR_PACKED"
    elif entropy >= 5.0:
        return "MEDIUM_STRUCTURED"
    elif entropy > 0.0:
        return "LOW_TEXT_OR_SPARSE"
    else:
        return "EMPTY"
