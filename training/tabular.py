"""Feature-only IDS2018 baseline; never invents graph endpoints or deploys weights."""
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from model_runtime import FEATURE_NAMES
from training.data import ALIASES, normalize, sha256, shards as graph_shards, read_metadata
from training.runner import metrics, choose_threshold
from training.assessment import assess_protocol

KNOWN = {normalize(x) for x in ['Benign','Bot','Brute Force -Web','Brute Force -XSS','SQL Injection','Infilteration','Infiltration','DDOS attack-HOIC','DDOS attack-LOIC-UDP','DDoS attacks-LOIC-HTTP','DoS attacks-GoldenEye','DoS attacks-Slowloris','DoS attacks-SlowHTTPTest','DoS attacks-Hulk','FTP-BruteForce','SSH-Bruteforce']}


def save(path, value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def feature_columns(headers):
    index={}
    for name in headers:
        index.setdefault(normalize(name),[]).append(name)
    result=[]
    for aliases in ALIASES:
        matches={n for a in aliases for n in index.get(normalize(a),[])}
        if len(matches)!=1: raise ValueError(f'Missing/ambiguous feature: {aliases[0]}')
        result.append(matches.pop())
    if len(index.get('label',[]))!=1: raise ValueError('Missing/ambiguous label')
    return result,index['label'][0]


def prepare(manifest, output):
    manifest=Path(manifest).resolve();output=Path(output);config=json.loads(manifest.read_text())
    sources=config['sources'];groups=[s['group'] for s in sources]
    if len(groups)!=len(set(groups)) or {s['split'] for s in sources}!={'train','validation'}:
        raise ValueError('Unique capture groups and train/validation splits required')
    output.mkdir(parents=True,exist_ok=False)
    metadata=dict(version=1,model_type='feature_only',features=FEATURE_NAMES,manifest=config,shards=[],sources=[])
    hashes=set()
    for i,source in enumerate(sources):
        path=(manifest.parent/source['path']).resolve()
        if path.suffix!='.csv': raise ValueError('Temporary or non-CSV inputs forbidden')
        digest=sha256(path)
        if digest in hashes: raise ValueError('Duplicate source file across capture groups')
        hashes.add(digest)
        columns,label=feature_columns(pd.read_csv(path,nrows=0).columns)
        stats=dict(group=source['group'],split=source['split'],source_sha256=digest,rows=0,accepted=0,rejected=0,labels={})
        with (output/f'rejected-{i:02d}.jsonl').open('w') as rejected:
            for j,frame in enumerate(pd.read_csv(path,usecols=[*columns,label],dtype=str,keep_default_na=False,chunksize=50000)):
                raw=frame[columns].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=np.float64)
                names=frame[label].str.strip().to_numpy()
                normalized=np.array([normalize(n) for n in names])
                repeated=normalized=='label'
                unknown=set(normalized)-KNOWN-{'label'}
                if unknown: raise ValueError(f'Unknown explicit labels in {path}: {sorted(unknown)}')
                valid=np.isfinite(raw).all(axis=1)&(raw>=0).all(axis=1)&~repeated
                transformed=np.log1p(raw[valid])
                if (transformed>100).any(): raise ValueError('Feature transform exceeds supported range')
                for offset in np.flatnonzero(~valid):
                    rejected.write(json.dumps(dict(record=stats['rows']+int(offset)+2,label=str(names[offset]),reason='repeated_header' if repeated[offset] else 'nonfinite_or_negative_feature'))+'\n')
                stats['rows']+=len(frame);stats['accepted']+=int(valid.sum());stats['rejected']+=int((~valid).sum())
                for n,count in zip(*np.unique(names[valid],return_counts=True)):
                    stats['labels'][str(n)]=stats['labels'].get(str(n),0)+int(count)
                if valid.any():
                    name=f'{i:02d}-{j:05d}.npz'
                    np.savez(output/name,msg=transformed.astype(np.float32),label=(normalized[valid]!='benign').astype(np.int64),family=names[valid].astype(str))
                    metadata['shards'].append(dict(path=name,sha256=sha256(output/name),rows=int(valid.sum()),group=source['group'],split=source['split']))
        if stats['accepted']==0: raise ValueError(f'No valid rows in {path}')
        if digest!=sha256(path): raise ValueError('Source changed while being read')
        metadata['sources'].append(stats)
        save(output/'preparation-progress.json',metadata)
        print(json.dumps(dict(event='prepared_source',**stats)),flush=True)
    save(output/'dataset.json',metadata)


def iter_data(data,graph_data,split,shuffle=False,seed=42,training_source='combined',include_family=False):
    metadata=json.loads((Path(data)/'dataset.json').read_text())
    if metadata['features']!=FEATURE_NAMES: raise ValueError('Feature schema mismatch')
    selected=[dict(s,root=str(data)) for s in metadata['shards'] if s['split']==split]
    if split=='train' and training_source=='cic2017':selected=[]
    selected.extend(dict(s,root=str(graph_data)) for s in read_metadata(graph_data)['shards'] if s['split']==split)
    if shuffle: np.random.default_rng(seed).shuffle(selected)
    for shard in selected:
        path=Path(shard['root'])/shard['path']
        if sha256(path)!=shard['sha256']: raise ValueError('Shard integrity failure')
        with np.load(path,allow_pickle=False) as f:
            if include_family:
                family=(np.array([normalize(str(n)) for n in f['family']]) if 'family' in f else
                        np.where(f['label']==1,'cic2017_attack','benign'))
                yield shard['group'],f['msg'],f['label'],family
            else:
                yield shard['group'],f['msg'],f['label']


def family_weights(counts,cap=20.):
    """Training-only capped weighting; no unverified family inference for CIC2017."""
    attack_counts=[n for name,n in counts.items() if name!='benign']
    largest=max(attack_counts,default=1)
    return {name:1. if name=='benign' else min(cap,math.sqrt(largest/n)) for name,n in counts.items()}


class FeatureModel(nn.Module):
    def __init__(self,hidden=64):
        super().__init__();self.hidden=hidden
        self.register_buffer('mean',torch.zeros(16));self.register_buffer('scale',torch.ones(16))
        self.net=nn.Sequential(nn.Linear(16,hidden),nn.ReLU(),nn.Linear(hidden,1)) if hidden else nn.Linear(16,1)
    def forward(self,msg):
        return self.net((msg-self.mean)/self.scale.clamp_min(.001)).squeeze(-1)


@torch.no_grad()
def predict(model,data,graph_data,split):
    model.eval();ys=[];ps=[];gs=[];names=[];start=time.perf_counter()
    for group,x,y in iter_data(data,graph_data,split):
        if group not in names: names.append(group)
        scores=[]
        for i in range(0,len(y),4096): scores.append(model(torch.from_numpy(x[i:i+4096])).sigmoid().numpy())
        ys.append(y);ps.append(np.concatenate(scores));gs.append(np.full(len(y),names.index(group),dtype=np.int16))
    return dict(y=np.concatenate(ys),p=np.concatenate(ps),g=np.concatenate(gs),groups=names,elapsed=time.perf_counter()-start)


def report(prediction,threshold):
    result=metrics(prediction['y'],prediction['p'],threshold)
    result['per_capture']={name:metrics(prediction['y'][prediction['g']==i],prediction['p'][prediction['g']==i],threshold) for i,name in enumerate(prediction['groups'])}
    result['elapsed_seconds']=prediction['elapsed']
    return result


def train(args):
    if args.epochs<1 or args.patience<1 or args.hidden<0 or args.threads<1: raise ValueError('Invalid training sizes')
    torch.set_num_threads(args.threads);torch.manual_seed(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    model=FeatureModel(args.hidden)
    training_source=getattr(args,'training_source','combined')
    balance_families=getattr(args,'family_weighting',False)
    count=0;mean=np.zeros(16);m2=np.zeros(16);classes=np.zeros(2,dtype=np.int64)
    family_counts={}
    for _,x,y,family in iter_data(args.data,args.graph_data,'train',training_source=training_source,include_family=True):
        values=x.astype(np.float64);n=len(y);delta=values.mean(0)-mean
        m2+=((values-values.mean(0))**2).sum(0)+delta**2*count*n/(count+n)
        mean+=delta*n/(count+n);count+=n;classes+=np.bincount(y,minlength=2)
        for name,n in zip(*np.unique(family,return_counts=True)):family_counts[str(name)]=family_counts.get(str(name),0)+int(n)
    if np.any(classes==0): raise ValueError('Both training classes required')
    model.mean.copy_(torch.tensor(mean,dtype=torch.float32));model.scale.copy_(torch.tensor(np.maximum(np.sqrt(m2/count),.001),dtype=torch.float32))
    weight=min(20.,math.sqrt(classes[0]/classes[1]));criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight),reduction='none')
    weights=family_weights(family_counts) if balance_families else {name:1. for name in family_counts}
    mean_weight=sum(family_counts[name]*w for name,w in weights.items())/count
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    provenance=dict(model_type='feature_only_mlp' if args.hidden else 'logistic_regression',features=FEATURE_NAMES,hidden=args.hidden,
        data_sha256=sha256(Path(args.data)/'dataset.json'),graph_data_sha256=sha256(Path(args.graph_data)/'dataset.json'),positive_weight=weight,
        training_counts=classes.tolist(),seed=args.seed,torch_version=str(torch.__version__),args=vars(args),deployment_approved=False)
    provenance.update(training_source=training_source,family_counts=family_counts,family_weights=weights,family_weight_mean=mean_weight)
    save(output/'run.json',provenance);best=None;stale=0;history=[]
    for epoch in range(args.epochs):
        model.train();total=0;loss_sum=0.;began=time.perf_counter();rng=np.random.default_rng(args.seed+epoch)
        for group,x,y,family in iter_data(args.data,args.graph_data,'train',True,args.seed+epoch,training_source=training_source,include_family=True):
            example_weights=np.array([weights[str(name)]/mean_weight for name in family],dtype=np.float32)
            indices=rng.permutation(len(y))
            for start in range(0,len(y),1024):
                idx=indices[start:start+1024];optimizer.zero_grad(set_to_none=True)
                loss=(criterion(model(torch.from_numpy(x[idx])),torch.tensor(y[idx],dtype=torch.float32))*torch.from_numpy(example_weights[idx])).mean()
                if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=len(idx);loss_sum+=float(loss.detach())*len(idx)
        validation=predict(model,args.data,args.graph_data,'validation')
        threshold,feasible=choose_threshold(validation['y'],validation['p'],.95,.001,.95)
        result=report(validation,threshold);checks=assess_protocol(result,.95,.95,.001)
        score=(int(feasible and checks['empirical_targets_met']),result['recall'] if feasible else 0.,result['average_precision'],-result['log_loss'])
        entry=dict(epoch=epoch+1,loss=loss_sum/total,elapsed_seconds=time.perf_counter()-began,
            validation_average_precision=result['average_precision'],validation_target_met=bool(feasible and checks['empirical_targets_met']),threshold=threshold)
        history.append(entry);save(output/'history.json',history);print(json.dumps(entry),flush=True)
        if best is None or score>best:
            best=score;stale=0
            torch.save(dict(**provenance,epoch=epoch+1,state=model.state_dict(),threshold=threshold,validation_target_met=entry['validation_target_met']),output/'best.pt')
            save(output/'validation.json',result)
            save(output/'validation-at-0.5.json',report(validation,.5))
        else: stale+=1
        if stale>=args.patience: break
    checkpoint=torch.load(output/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(checkpoint['state'])
    if getattr(args,'validation_only',False):
        print(json.dumps(dict(event='completed_validation_only',model_type=provenance['model_type'],deployment_approved=False)),flush=True)
        return
    # Frozen Friday CIC2017 test, same prepared rows used by the graph candidate.
    test=predict(model,args.data,args.graph_data,'test');result=report(test,checkpoint['threshold'])
    result['target_checks']=assess_protocol(result,.95,.95,.001)
    result['validation_target_met']=checkpoint['validation_target_met'];result['deployment_approved']=False
    result['checkpoint_sha256']=sha256(output/'best.pt')
    save(output/'test-report.json',result)
    print(json.dumps(dict(event='completed',model_type=provenance['model_type'],deployment_approved=False)),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--manifest',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('train');p.add_argument('--data',required=True);p.add_argument('--graph-data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--patience',type=int,default=3);p.add_argument('--hidden',type=int,default=64);p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--training-source',choices=['combined','cic2017'],default='combined')
    p.add_argument('--family-weighting',action='store_true')
    p.add_argument('--validation-only',action='store_true',help='Do not open held-out test data during controlled comparisons')
    args=parser.parse_args()
    if args.command=='prepare':prepare(args.manifest,args.output)
    else:train(args)

if __name__=='__main__':main()
