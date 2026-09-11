"""Feature Extraction & Threat Modeling Engine."""
from .entropy import shannon_entropy, classify_entropy
from .handshake import HandshakeMetrics
from .port_roles import classify_port_roles, PortProfile
from .flow_aggregator import FlowRecord, FlowAggregator

__all__ = [
    "shannon_entropy",
    "classify_entropy",
    "HandshakeMetrics",
    "classify_port_roles",
    "PortProfile",
    "FlowRecord",
    "FlowAggregator",
]
