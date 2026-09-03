import pandas as pd
import numpy as np
from typing import Iterator
from .schema import NetworkEvent

class CICIDSLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> Iterator[NetworkEvent]:
        chunksize = 10000
        for chunk in pd.read_csv(self.filepath, chunksize=chunksize, encoding='utf-8', on_bad_lines='skip'):
            chunk.columns = chunk.columns.str.strip().str.lower()
            chunk.replace([np.inf, -np.inf], np.nan, inplace=True)
            chunk.dropna(inplace=True)
            
            if 'dst port' in chunk.columns:
                chunk = chunk[chunk['dst port'] != 'Dst Port']
            
            for _, row in chunk.iterrows():
                yield NetworkEvent(
                    timestamp=pd.to_datetime(row.get('timestamp', 0)).timestamp() if 'timestamp' in row else 0.0,
                    src_ip=str(row.get('src ip', '0.0.0.0')),
                    dst_ip=str(row.get('dst ip', '0.0.0.0')),
                    src_port=int(row.get('src port', 0)),
                    dst_port=int(row.get('dst port', 0)),
                    protocol=str(row.get('protocol', 'tcp')),
                    packet_size=int(row.get('totlen fwd pkts', 0) or 0),
                    label=str(row.get('label', 'Benign'))
                )

class CTU13Loader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self) -> Iterator[NetworkEvent]:
        chunksize = 10000
        for chunk in pd.read_csv(self.filepath, chunksize=chunksize):
            chunk.columns = chunk.columns.str.strip().str.lower()
            for _, row in chunk.iterrows():
                sport = row.get('sport', '0')
                dport = row.get('dport', '0')
                try:
                    sport = int(sport, 16) if isinstance(sport, str) and sport.startswith('0x') else int(sport)
                except ValueError:
                    sport = 0
                try:
                    dport = int(dport, 16) if isinstance(dport, str) and dport.startswith('0x') else int(dport)
                except ValueError:
                    dport = 0
                    
                yield NetworkEvent(
                    timestamp=pd.to_datetime(row.get('starttime', 0)).timestamp() if 'starttime' in row else 0.0,
                    src_ip=str(row.get('srcaddr', '0.0.0.0')),
                    dst_ip=str(row.get('dstaddr', '0.0.0.0')),
                    src_port=sport,
                    dst_port=dport,
                    protocol=str(row.get('proto', 'tcp')),
                    packet_size=int(row.get('totbytes', 0)),
                    label=str(row.get('label', 'Normal'))
                )
