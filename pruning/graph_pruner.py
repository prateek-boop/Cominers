import yaml
from typing import Iterator
from ingestion.schema import NetworkEvent

class GraphPruner:
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.drop_protocols = set(self.config.get('drop_protocols', []))
        self.keep_ports = set(self.config.get('keep_ports', []))

    def prune(self, events: Iterator[NetworkEvent]) -> Iterator[NetworkEvent]:
        for event in events:
            if event.protocol.lower() in self.drop_protocols:
                continue
                
            if event.dst_port in self.keep_ports or event.src_port in self.keep_ports:
                yield event
                continue
                
            if event.protocol.lower() == 'tcp':
                yield event
                continue
                
            yield event
