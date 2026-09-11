"""Checkpoint-backed CyberTGN inference shared by the CLI, API and evaluator."""
from pathlib import Path
import hashlib
import threading

import numpy as np
import torch
from torch_geometric.nn.models.tgn import LastNeighborLoader
from models.current_flow import load_current_flow, combined_logits, encode_timestamps
from models.memory import MemoryModule
from models.tgn import CyberTGN
from models.multitask_heads import AttackProbabilityHead, MitreStageClassifier

ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = ROOT / 'checkpoints/tgn_best.pt'
FEATURE_NAMES = [
    'Flow Duration', 'Total Fwd Packet', 'Total Bwd packets',
    'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Fwd IAT Mean', 'Bwd IAT Mean',
    'Fwd Packets/s', 'Bwd Packets/s', 'Average Packet Size', 'Down/Up Ratio',
]
STAGES = ['benign', 'reconnaissance', 'initial_access', 'credential_access',
          'lateral_movement', 'command_and_control', 'exfiltration', 'impact']


class ModelRuntime:
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, threads=2):
        torch.set_num_threads(threads)
        self.path = Path(checkpoint).resolve()
        checkpoint = torch.load(self.path, map_location='cpu', weights_only=True)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.metadata = {k: v for k, v in checkpoint.items() if isinstance(v, (int, float, str))}
        mem = checkpoint['memory']
        self.num_nodes, dim = mem['memory.memory'].shape
        time_dim = mem['memory.time_enc.lin.weight'].shape[0]
        self.msg_dim = mem['memory.gru.weight_ih'].shape[1] - 2 * dim - time_dim
        embedding_dim = checkpoint['prob_head']['mlp.0.weight'].shape[1] // 2
        self.num_classes = checkpoint['mitre_head']['mlp.2.weight'].shape[0]
        time_transform = checkpoint.get('model_config',{}).get('time_transform','identity')
        self.memory = MemoryModule(self.num_nodes, self.msg_dim, dim, time_dim,time_transform)
        self.gnn = CyberTGN(self.memory, dim, embedding_dim, self.msg_dim, time_dim,time_transform)
        self.prob_head = AttackProbabilityHead(embedding_dim)
        self.stage_head = MitreStageClassifier(embedding_dim, self.num_classes)
        self.memory.load_state_dict(mem, strict=True)
        self.gnn.load_state_dict(checkpoint['gnn'], strict=True)
        self.prob_head.load_state_dict(checkpoint['prob_head'], strict=True)
        self.stage_head.load_state_dict(checkpoint['mitre_head'], strict=True)
        self.current_flow_head = load_current_flow(checkpoint)
        self.timestamp_encoding = checkpoint.get("timestamp_encoding", "legacy_float32")
        self.stage_training_counts = checkpoint.get("stage_training_counts")
        for module in (self.memory, self.gnn, self.prob_head, self.stage_head):
            module.eval()
        if any(not torch.isfinite(p).all() for m in (self.gnn, self.prob_head, self.stage_head) for p in m.parameters()):
            raise ValueError('Checkpoint contains non-finite weights')
        self.neighbors = LastNeighborLoader(self.num_nodes, size=10, device='cpu')
        self.assoc = torch.empty(self.num_nodes, dtype=torch.long)
        self.lock = threading.Lock()
        self.reset(1)

    def reset(self, capacity):
        # Saved memory has validation history and no matching saved neighbor graph.
        self.memory.reset_state()
        self.neighbors.reset_state()
        self.history_t = torch.empty(capacity, dtype=torch.long)
        self.history_msg = torch.empty((capacity, self.msg_dim), dtype=torch.float32)
        self.cursor = 0

    @torch.no_grad()
    def predict_batch(self, src, dst, t, msg):
        """Score from prior history, then update state; events in a batch share history."""
        end = self.cursor + len(src)
        if end > len(self.history_t):
            raise ValueError('Sequence exceeds allocated history capacity')
        nodes, edges, ids = self.neighbors(torch.cat([src, dst]).unique())
        self.assoc[nodes] = torch.arange(len(nodes))
        z = self.gnn(nodes, edges, self.history_t[ids], self.history_msg[ids])
        a, b = z[self.assoc[src]], z[self.assoc[dst]]
        attack_logits, stage_logits = combined_logits(self.prob_head, self.stage_head, self.current_flow_head, a, b, msg)
        probabilities = attack_logits.sigmoid()
        stages = stage_logits.softmax(-1)
        if not torch.isfinite(probabilities).all() or not torch.isfinite(stages).all():
            raise ValueError('Model produced non-finite predictions')
        self.memory.update_state(src, dst, t.long(), msg)
        self.neighbors.insert(src, dst)
        self.history_t[self.cursor:end] = t.long()
        self.history_msg[self.cursor:end] = msg
        self.cursor = end
        return probabilities.numpy(), stages.numpy()

    def predict_sequence(self, flows, feature_format='raw', threshold=0.5, batch_size=200):
        """Independent, ordered sequence. No state is shared across API requests."""
        if not isinstance(batch_size, int) or not 1 <= batch_size <= 200:
            raise ValueError('batch_size must be between 1 and 200')
        if feature_format not in ('raw', 'signed_log1p'):
            raise ValueError('Unknown feature format')
        if not flows or len(flows) > 1000:
            raise ValueError('Provide between 1 and 1000 flows')
        if not 0 <= threshold <= 1:
            raise ValueError('threshold must be between 0 and 1')
        times = np.asarray([f['timestamp'] for f in flows], dtype=np.float64)
        features = np.asarray([f['features'] for f in flows], dtype=np.float64)
        if features.shape != (len(flows), self.msg_dim) or not np.isfinite(features).all():
            raise ValueError(f'Each flow must contain {self.msg_dim} finite features')
        if not np.isfinite(times).all() or np.any(times < 0) or np.any(times > 1e12) or np.any(np.diff(times) < 0):
            raise ValueError('Timestamps must be finite, nonnegative and in ascending order (maximum 1e12)')
        if feature_format == 'raw':
            features = np.sign(features) * np.log1p(np.abs(features))
        if np.any(np.abs(features) > 100):
            raise ValueError('Transformed feature magnitude exceeds supported range (100)')
        mapping = {}
        def node(ip):
            if ip not in mapping:
                mapping[ip] = len(mapping)
            return mapping[ip]
        src = torch.tensor([node(f['src_ip']) for f in flows])
        dst = torch.tensor([node(f['dst_ip']) for f in flows])
        if len(mapping) > self.num_nodes:
            raise ValueError('Node capacity exceeded')
        # Match training timestamp conversion, including float32 rounding before int64.
        t = encode_timestamps(times, self.timestamp_encoding)
        msg = torch.tensor(features, dtype=torch.float32)
        output = []
        with self.lock:
            self.reset(len(flows))
            for start in range(0, len(flows), batch_size):
                end = min(start + batch_size, len(flows))
                p, stages = self.predict_batch(src[start:end], dst[start:end], t[start:end], msg[start:end])
                for offset, i in enumerate(range(start, end)):
                    stage = int(stages[offset].argmax())
                    output.append(dict(index=i, attack_probability=float(p[offset]),
                        is_attack=bool(p[offset] >= threshold), mitre_stage=stage,
                        stage_name=STAGES[stage] if stage < len(STAGES) else f'unmapped_{stage}',
                        stage_probabilities=stages[offset].tolist(), cold_start=(start == 0)))
        return output
