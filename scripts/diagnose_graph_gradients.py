"""Matched-seed gradient probe on training rows only; no checkpoint promotion."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from training.runner import TrainableTGN
from training.data import read_metadata,shards
root=Path(__file__).resolve().parents[1];data=root/'data/cic2017-prepared';meta=read_metadata(data)
_,batch=next(shards(data,'train'));torch.set_num_threads(2)
results=[]
for transform in ('identity','signed_log1p'):
 torch.manual_seed(42)
 model=TrainableTGN(meta['max_nodes'],128,128,32,64,100000,time_transform=transform)
 model.current_flow_head.mean.copy_(torch.tensor(meta['train_mean']));model.current_flow_head.scale.copy_(torch.tensor(meta['train_scale']))
 model.train();model.reset();optimizer=torch.optim.AdamW(model.parameters(),lr=.0003)
 maxima={};norms=[];losses=[]
 for start in range(0,4096,32):
  end=start+32;src=torch.tensor(batch['src'][start:end]);dst=torch.tensor(batch['dst'][start:end]);t=torch.tensor(batch['timestamp'][start:end],dtype=torch.long);msg=torch.tensor(batch['msg'][start:end]);y=torch.tensor(batch['label'][start:end],dtype=torch.float32)
  optimizer.zero_grad(set_to_none=True);a,_=model.score(src,dst,msg);loss=torch.nn.functional.binary_cross_entropy_with_logits(a,y)
  model.update(src,dst,t,msg);loss.backward()
  for name,p in model.named_parameters():
   if p.grad is not None:maxima[name]=max(maxima.get(name,0),float(p.grad.norm()))
  norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);norms.append(float(norm));losses.append(float(loss.detach()));optimizer.step();model.memory.detach()
 result=dict(transform=transform,rows=4096,seed=42,max_gradient_norm=max(norms),median_gradient_norm=float(np.median(norms)),clipped_steps=sum(n>1 for n in norms),steps=len(norms),mean_loss=float(np.mean(losses)),largest_parameter_gradients=sorted(maxima.items(),key=lambda p:-p[1])[:5])
 results.append(result);print(json.dumps(result),flush=True)
p=root/'reports/audit/graph-gradient-probe.json';p.write_text(json.dumps(dict(scope='First 4096 training rows only, matched initialization and optimizer; diagnostic, not accuracy evaluation',results=results),indent=2))
