import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
from torch_geometric.nn.models.tgn import TimeEncoder
from models.time_features import TransformedTimeEncoder
from training.runner import TrainableTGN
from model_runtime import ModelRuntime
from engine.streaming_runtime import StreamingModelRuntime
from engine.stateful_tgn import StatefulCyberTGN
from graph.graph_formatter import FormattedGraphBatch
from features.flow_aggregator import FlowRecord
from training.tabular import family_weights

class TrainingStabilityTests(unittest.TestCase):
    def test_identity_time_encoding_matches_original(self):
        original=TimeEncoder(4);wrapped=TransformedTimeEncoder(original)
        t=torch.tensor([0.,1.,-100.,1.5e9])
        torch.testing.assert_close(original(t),wrapped(t),rtol=0,atol=0)
        self.assertEqual(set(original.state_dict()),set(wrapped.state_dict()))
    def test_log_time_gradient_bound(self):
        encoder=TransformedTimeEncoder(TimeEncoder(4),'signed_log1p')
        encoder(torch.tensor([1.5e9])).sum().backward()
        self.assertLessEqual(float(encoder.lin.weight.grad.abs().max()),np.log1p(1.5e9)+1e-5)
    def test_family_weights_cap_and_benign_reference(self):
        w=family_weights({'benign':10000,'common':4000,'rare':1})
        self.assertEqual(w,{'benign':1.,'common':1.,'rare':20.})
    def test_scaled_time_checkpoint_matches_all_runtimes(self):
        torch.set_num_threads(2);torch.manual_seed(3)
        model=TrainableTGN(16,16,16,8,16,32,time_transform='signed_log1p');model.eval()
        flows=[dict(src_ip='192.0.2.1',dst_ip='192.0.2.2',timestamp=1700000000+i,features=[float(i+1)]*16) for i in range(8)]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'candidate.pt';torch.save(model.checkpoint({}),path)
            restored,_=TrainableTGN.restore(path)
            baseline=ModelRuntime(path).predict_sequence(flows,batch_size=2)
            stream=StreamingModelRuntime(path,max_history=32).score_flows(flows,batch_size=2)
            expected=[r['attack_probability'] for r in baseline]
            np.testing.assert_allclose(expected,[r['attack_probability'] for r in stream],atol=1e-6)
            stateful=StatefulCyberTGN(path,history_capacity=32);actual=[];trainer=[]
            for i in range(0,8,2):
                chunk=flows[i:i+2];src=torch.tensor([0,0]);dst=torch.tensor([1,1]);t=torch.tensor([f['timestamp'] for f in chunk]);msg=torch.tensor(np.log1p([f['features'] for f in chunk]),dtype=torch.float32)
                with torch.no_grad():trainer.extend(restored.score(src,dst,msg)[0].sigmoid().tolist());restored.update(src,dst,t,msg)
                records=[FlowRecord(str(i+j),f['src_ip'],f['dst_ip'],1,2,'tcp',f['timestamp'],f['timestamp']) for j,f in enumerate(chunk)]
                r,_=stateful.process_graph_batch(FormattedGraphBatch(src,dst,t,msg,records,np.array([f['features'] for f in chunk])))
                actual.extend(row['attack_probability'] for row in r)
            np.testing.assert_allclose(expected,trainer,atol=1e-6)
            np.testing.assert_allclose(expected,actual,atol=1e-6)
