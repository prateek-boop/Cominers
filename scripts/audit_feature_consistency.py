"""Bounded stratified train/validation audit; never reads test shards."""
import json,sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.tabular import iter_data
from model_runtime import FEATURE_NAMES
root=Path(__file__).resolve().parents[1];rng=np.random.default_rng(42);samples={};totals=Counter()
for split in ('train','validation'):
 for group,x,y,family in iter_data(root/'data/ids2018-tabular',root/'data/cic2017-prepared',split,include_family=True):
  for name in np.unique(family):
   mask=family==name;values=x[mask];labels=y[mask];key=(split,group,str(name));totals[key]+=len(values)
   keys=rng.random(len(values));old=samples.get(key)
   if old is not None:keys=np.r_[old[0],keys];values=np.concatenate([old[1],values]);labels=np.r_[old[2],labels]
   keep=np.argpartition(keys,511)[:512] if len(keys)>512 else np.arange(len(keys))
   samples[key]=(keys[keep],values[keep],labels[keep])
records=[];fingerprints={};contexts={}
for (split,group,family),(_,x,y) in samples.items():
 raw=np.expm1(x.astype(np.float64));errors={}
 comparisons={'forward_mean':(raw[:,3],raw[:,1],raw[:,5]),'backward_mean':(raw[:,4],raw[:,2],raw[:,6]),'flow_bytes_rate':((raw[:,3]+raw[:,4])*1e6,raw[:,0],raw[:,7]),'flow_packets_rate':((raw[:,1]+raw[:,2])*1e6,raw[:,0],raw[:,8])}
 for name,(numerator,denominator,observed) in comparisons.items():
  valid=denominator>0;expected=numerator[valid]/denominator[valid]
  mismatch=~np.isclose(expected,observed[valid],rtol=.01,atol=.01)
  errors[name]=dict(eligible=int(valid.sum()),mismatches=int(mismatch.sum()))
 for row,label in zip(x,y):
  fingerprint=row.tobytes()
  fingerprints.setdefault(fingerprint,set()).add(int(label))
  contexts.setdefault(fingerprint,{}).setdefault((split,group),set()).add(int(label))
 records.append(dict(split=split,group=group,family=family,population_rows=totals[(split,group,family)],sample_rows=len(y),feature_medians=np.median(raw,axis=0).tolist(),algebra_checks=errors))
report=dict(scope='Uniform priority reservoir, up to 512 rows per family/capture/split, seed 42. Train and validation only. No label truth verification or blanket row correction.',feature_order=FEATURE_NAMES,records=records,distinct_sample_feature_vectors=len(fingerprints),sample_feature_vectors_with_conflicting_binary_labels=sum(len(s)>1 for s in fingerprints.values()),interpretation='Algebra checks use 1% relative / 0.01 absolute tolerance after reversing float32 log1p; mismatches indicate exporter semantics or data issues to investigate, not proof labels are wrong. Identical 16-feature vectors cannot be distinguished by a deterministic feature-only classifier.')
within_capture=Counter()
for captures in contexts.values():
 for key,labels in captures.items():
  if len(labels)>1:within_capture[key]+=1
report['within_capture_conflicting_sample_vectors']=[dict(split=split,group=group,vectors=count) for (split,group),count in sorted(within_capture.items())]
(root/'reports/audit/feature-consistency.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ('records','feature_order')}),flush=True)
