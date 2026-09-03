from typing import Dict
from .schema import NetworkEvent

class Normalizer:
    def __init__(self):
        self.ip_mapping: Dict[str, int] = {}
        self.next_node_id = 0
        self.protocol_map = {'tcp': 6, 'udp': 17, 'icmp': 1, 'other': 0}

    def get_node_id(self, ip: str) -> int:
        if ip not in self.ip_mapping:
            self.ip_mapping[ip] = self.next_node_id
            self.next_node_id += 1
        return self.ip_mapping[ip]

    def normalize(self, event: NetworkEvent) -> dict:
        return {
            'timestamp': event.timestamp,
            'src_node_id': self.get_node_id(event.src_ip),
            'dst_node_id': self.get_node_id(event.dst_ip),
            'src_port': event.src_port,
            'dst_port': event.dst_port,
            'protocol_encoded': self.protocol_map.get(event.protocol.lower(), 0),
            'packet_size': event.packet_size,
            'label': event.label
        }
