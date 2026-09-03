import numpy as np
from ingestion.schema import NetworkEvent

class PacketFeatureExtractor:
    def extract(self, event: NetworkEvent) -> np.ndarray:
        flags = event.tcp_flags or 0
        syn = (flags & 0x02) >> 1
        ack = (flags & 0x10) >> 4
        rst = (flags & 0x04) >> 2
        fin = flags & 0x01
        psh = (flags & 0x08) >> 3
        
        return np.array([
            float(event.packet_size),
            float(event.ttl or 0),
            float(event.payload_entropy or 0.0),
            float(event.tcp_window or 0),
            float(syn),
            float(ack),
            float(rst),
            float(fin),
            float(psh)
        ], dtype=np.float32)
