"""Supervised temporal training with separate validation, calibration and test."""
import argparse
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch_geometric.nn.models.tgn import LastNeighborLoader
from models.memory import MemoryModule
from models.tgn import CyberTGN
from models.multitask_heads import AttackProbabilityHead, MitreStageClassifier
from models.current_flow import CurrentFlowHead, combined_logits, encode_timestamps
from model_runtime import FEATURE_NAMES, STAGES
from training.data import prepare, read_metadata, shards, sha256
from training.assessment import additional_metrics, assess_protocol
from training.losses import supervised_loss


class TrainableTGN(nn.Module):
    def __init__(self, num_nodes, memory_dim=128, embedding_dim=128, time_dim=32, direct_dim=64,
                 history_capacity=100000, device='cpu', time_transform='identity'):
        super().__init__()
        self.config = dict(num_nodes=num_nodes,memory_dim=memory_dim,embedding_dim=embedding_dim,
                           time_dim=time_dim,direct_dim=direct_dim,history_capacity=history_capacity,time_transform=time_transform)
        self.memory = MemoryModule(num_nodes,16,memory_dim,time_dim,time_transform)
        self.gnn = CyberTGN(self.memory,memory_dim,embedding_dim,16,time_dim,time_transform)
        self.prob_head = AttackProbabilityHead(embedding_dim)
        self.stage_head = MitreStageClassifier(embedding_dim,8)
        self.current_flow_head = CurrentFlowHead(16,direct_dim,8)
        self.to(device)
        self.neighbors = LastNeighborLoader(num_nodes,size=10,device=device)
        self.assoc = torch.empty(num_nodes,dtype=torch.long,device=device)
        self.history_t = torch.empty(history_capacity,dtype=torch.long,device=device)
        self.history_msg = torch.empty((history_capacity,16),device=device)
        self.cursor = 0

    @property
    def device(self):
        return self.current_flow_head.mean.device

    def reset(self):
        self.memory.reset_state()
        self.neighbors.reset_state()
        self.cursor = 0

    def score(self, src, dst, msg):
        if not 1 <= len(src) <= len(self.history_t):
            raise ValueError('Batch must be nonempty and fit history capacity')
        if self.cursor + len(src) > len(self.history_t):
            self.neighbors.reset_state()
            self.cursor = 0
        nodes, edges, ids = self.neighbors(torch.cat([src,dst]).unique())
        self.assoc[nodes] = torch.arange(len(nodes),device=self.device)
        z = self.gnn(nodes,edges,self.history_t[ids],self.history_msg[ids])
        return combined_logits(self.prob_head,self.stage_head,self.current_flow_head,
                               z[self.assoc[src]],z[self.assoc[dst]],msg)

    def update(self, src, dst, timestamp, msg):
        self.memory.update_state(src,dst,timestamp,msg)
        self.neighbors.insert(src,dst)
        end = self.cursor + len(src)
        self.history_t[self.cursor:end] = timestamp
        self.history_msg[self.cursor:end] = msg
        self.cursor = end

    def checkpoint(self, metadata):
        # Zero deployment memory; training hosts must never leak into inference.
        self.reset()
        cpu = lambda module: {k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
        return dict(memory=cpu(self.memory),gnn=cpu(self.gnn),prob_head=cpu(self.prob_head),
                    mitre_head=cpu(self.stage_head),current_flow_head=cpu(self.current_flow_head),
                    model_config=self.config,architecture_version=2,timestamp_encoding='int64_seconds',
                    feature_names=FEATURE_NAMES,stage_names=STAGES,**metadata)

    @classmethod
    def restore(cls, path, device='cpu'):
        checkpoint = torch.load(path,map_location='cpu',weights_only=True)
        if checkpoint.get('architecture_version') != 2:
            raise ValueError('This evaluator expects a v2 training checkpoint')
        model = cls(**checkpoint['model_config'],device=device)
        for attribute,key in [('memory','memory'),('gnn','gnn'),('prob_head','prob_head'),
                              ('stage_head','mitre_head'),('current_flow_head','current_flow_head')]:
            getattr(model,attribute).load_state_dict(checkpoint[key],strict=True)
        model.eval()
        model.reset()
        return model, checkpoint


def batches(directory, split, sizes, rng):
    """Batch complete captures; storage shard boundaries never change the protocol."""
    if not sizes or any(not isinstance(n, int) or n < 1 for n in sizes):
        raise ValueError('Positive batch sizes required')
    current, parts, buffered, target = None, [], 0, None
    for group, shard in shards(directory,split):
        if group != current:
            if parts:
                yield current, {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
            current, parts, buffered, target = group, [], 0, None
        start = 0
        while start < len(shard['label']):
            if target is None:
                target = rng.choice(sizes)
            end = min(start + target - buffered,len(shard['label']))
            parts.append({k:v[start:end] for k,v in shard.items()})
            buffered += end-start
            start = end
            if buffered == target:
                yield group, {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
                parts, buffered, target = [], 0, None
    if parts:
        yield current, {k:np.concatenate([p[k] for p in parts]) for k in parts[0]}


def tensors(model, batch):
    return (torch.as_tensor(batch['src'],device=model.device),
            torch.as_tensor(batch['dst'],device=model.device),
            encode_timestamps(batch['timestamp'],'int64_seconds').to(model.device),
            torch.as_tensor(batch['msg'],device=model.device))


def validate_predictions(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if (labels.ndim != 1 or probabilities.shape != labels.shape or not labels.size
            or not np.isin(labels, [0,1]).all() or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))):
        raise ValueError('Matching nonempty binary labels and finite probabilities in [0,1] required')
    return labels.astype(np.int64), probabilities


def metrics(labels, probabilities, threshold=.5):
    labels, probabilities = validate_predictions(labels, probabilities)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('Threshold must be finite and in [0,1]')
    predicted = probabilities >= threshold
    tp = int(np.sum(predicted & (labels==1)))
    fp = int(np.sum(predicted & (labels==0)))
    fn = int(np.sum(~predicted & (labels==1)))
    tn = int(np.sum(~predicted & (labels==0)))
    order = np.argsort(-probabilities,kind='stable')
    ys, ps = labels[order], probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(ps)),len(ps)-1]
    tps = np.cumsum(ys)[ends]
    fps = (ends + 1) - tps
    positives, negatives = int(labels.sum()), int((labels==0).sum())
    ap = float(np.sum(np.diff(np.r_[0,tps]) / positives * tps/(tps+fps))) if positives else None
    auc = float((np.trapezoid if hasattr(np, 'trapezoid') else np.trapz)(np.r_[0,tps]/positives,np.r_[0,fps]/negatives)) if positives and negatives else None
    return dict(events=len(labels),tp=tp,fp=fp,tn=tn,fn=fn,precision=tp/max(tp+fp,1),
                recall=tp/max(tp+fn,1),f1=2*tp/max(2*tp+fp+fn,1),false_positive_rate=fp/max(fp+tn,1),
                average_precision=ap,roc_auc=auc,threshold=threshold,
                brier=float(np.mean((probabilities-labels)**2)),
                **additional_metrics(labels,probabilities,tp,fp,tn,fn))


def choose_threshold(labels, probabilities, min_precision, max_fpr, min_recall=0.):
    labels, probabilities = validate_predictions(labels, probabilities)
    if not 0 <= min_precision <= 1 or not 0 <= max_fpr <= 1 or not 0 <= min_recall <= 1:
        raise ValueError('Operating-point targets must be in [0,1]')
    order = np.argsort(-probabilities,kind='stable')
    y,p = labels[order],probabilities[order]
    ends = np.r_[np.flatnonzero(np.diff(p)),len(p)-1]
    tp = np.cumsum(y)[ends]
    fp = ends+1-tp
    eligible = ((tp>0) & (tp/(tp+fp)>=min_precision)
                & (fp/max(int((labels==0).sum()),1)<=max_fpr)
                & (tp/max(int(labels.sum()),1)>=min_recall))
    if not eligible.any():
        return 1.0, False
    # Maximize recall while meeting validation precision and false-alarm targets.
    idx = np.flatnonzero(eligible)[np.argmax(tp[eligible])]
    return float(p[ends[idx]]), True


@torch.no_grad()
def predict(model, directory, split, batch_size, reset_events=0):
    model.eval()
    model.reset()
    current = None
    seen = 0
    labels, scores, stages, stage_preds, cold, groups = [],[],[],[],[],[]
    start_time = time.perf_counter()
    for group, batch in batches(directory,split,[batch_size],random.Random(0)):
        if group != current or (reset_events and seen >= reset_events):
            model.reset()
            current, seen = group, 0
        src,dst,t,msg = tensors(model,batch)
        logits, stage_logits = model.score(src,dst,msg)
        if not torch.isfinite(logits).all() or not torch.isfinite(stage_logits).all():
            raise ValueError('Model produced non-finite predictions')
        scores.extend(logits.sigmoid().cpu().tolist())
        labels.extend(batch['label'].tolist())
        stages.extend(batch['stage'].tolist())
        stage_preds.extend(stage_logits.argmax(-1).cpu().tolist())
        cold.extend([seen==0]*len(src))
        groups.extend([group]*len(src))
        model.update(src,dst,t,msg)
        seen += len(src)
    return dict(label=np.array(labels),probability=np.array(scores),stage=np.array(stages),
                stage_pred=np.array(stage_preds),cold=np.array(cold),group=np.array(groups),
                elapsed_seconds=time.perf_counter()-start_time)


def report_predictions(predictions, threshold):
    result = metrics(predictions['label'],predictions['probability'],threshold)
    result['elapsed_seconds'] = predictions['elapsed_seconds']
    result['per_capture'] = {str(g):metrics(predictions['label'][predictions['group']==g],
                                         predictions['probability'][predictions['group']==g],threshold)
                             for g in np.unique(predictions['group'])}
    mask = predictions['cold']
    result['cold_batch'] = metrics(predictions['label'][mask],predictions['probability'][mask],threshold) if mask.any() else None
    y,p = predictions['stage'],predictions['stage_pred']
    observed = np.unique(y[y>=0])
    result['stage_supervised_rows'] = int(np.sum(y>=0))
    result['stage_per_class_f1'] = {STAGES[int(k)]:float(2*np.sum((y==k)&(p==k))/max(1,np.sum(y==k)+np.sum((p==k)&(y>=0)))) for k in observed}
    result['stage_macro_f1_observed_labels'] = float(np.mean(list(result['stage_per_class_f1'].values()))) if len(observed) else None
    matrix = np.zeros((8,8),dtype=np.int64)
    np.add.at(matrix,(y[y>=0],p[y>=0]),1)
    result['stage_confusion_matrix'] = matrix.tolist()
    result['stage_class_order'] = STAGES
    result['stage_support'] = matrix.sum(axis=1).tolist()
    result['stage_per_class'] = {}
    for k,name in enumerate(STAGES):
        tp = int(matrix[k,k])
        support, predicted_support = int(matrix[k].sum()), int(matrix[:,k].sum())
        result['stage_per_class'][name] = dict(support=support,predicted_support=predicted_support,
            precision=tp/predicted_support if predicted_support else None,
            recall=tp/support if support else None,
            f1=2*tp/(support+predicted_support) if support+predicted_support else None)
    present = [v['f1'] for v in result['stage_per_class'].values() if v['f1'] is not None]
    result['stage_macro_f1_present_classes'] = float(np.mean(present)) if present else None
    result['stage_accuracy'] = float(np.trace(matrix)/matrix.sum()) if matrix.sum() else None
    return result


def train(args):
    validate_train_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        torch.cuda.manual_seed_all(args.seed)
    metadata = read_metadata(args.data)
    sizes = [int(n) for n in args.batch_sizes.split(',')]
    if any(n<1 or n>200 or n>args.history_capacity for n in sizes):
        raise ValueError('Batch sizes must be in [1,200] and fit history capacity')
    if args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0:
        raise ValueError('Positive epochs, patience and learning rate required')
    if args.node_capacity < metadata['max_nodes']:
        raise ValueError(f"--node-capacity must be at least {metadata['max_nodes']}")
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    model = TrainableTGN(args.node_capacity,args.memory_dim,args.embedding_dim,args.time_dim,args.direct_dim,
                         args.history_capacity,args.device,time_transform=getattr(args,'time_transform','identity'))
    model.current_flow_head.mean.copy_(torch.tensor(metadata['train_mean'],device=model.device))
    model.current_flow_head.scale.copy_(torch.tensor(metadata['train_scale'],device=model.device))
    counts = np.zeros(2,dtype=np.int64)
    stage_counts = np.zeros(8,dtype=np.int64)
    for _, shard in shards(args.data,'train'):
        counts += np.bincount(shard['label'],minlength=2)
        stage_counts += np.bincount(shard['stage'][shard['stage']>=0],minlength=8)
    if np.any(counts==0):
        raise ValueError('Training split must contain benign and attack examples')
    support = set()
    for _, shard in shards(args.data,'validation'):
        support.update(shard['label'].tolist())
    if support != {0,1}:
        raise ValueError('Validation requires both binary labels for meaningful evaluation')
    if not any(np.any(shard['label'] == 0) for _, shard in shards(args.data,'calibration')):
        raise ValueError('Calibration split requires held-out benign examples')
    weight_setting = getattr(args,'positive_weight','auto')
    positive_weight = min(20.,math.sqrt(counts[0]/counts[1])) if weight_setting=='auto' else float(weight_setting)
    positive_tensor = torch.tensor(positive_weight,dtype=torch.float32,device=model.device)
    weights = np.sqrt(max(stage_counts.max(),1)/np.maximum(stage_counts,1)).clip(1,10)
    stage_weights = torch.tensor(weights,dtype=torch.float32,device=model.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    best, stale, history = None,0,[]
    run = vars(args).copy()
    run.update(torch_version=str(torch.__version__),numpy_version=np.__version__,dataset_sha256=sha256(Path(args.data)/'dataset.json'),
               stage_training_counts=stage_counts.tolist(),feature_branch=True,
               binary_training_counts=counts.tolist(),resolved_positive_weight=positive_weight,
               resolved_stage_weights=weights.tolist(),min_recall=getattr(args,'min_recall',.95),
               selection_metric=getattr(args,'selection_metric','constrained_recall'))
    (output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    for epoch in range(args.epochs):
        model.train()
        model.reset()
        current, total_loss, total = None,0.,0
        binary_sum, stage_sum, stage_denominator_sum, gradient_max = 0.,0.,0.,0.
        for group,batch in batches(args.data,'train',sizes,random.Random(args.seed+epoch)):
            if current != group:
                model.reset()
                current = group
            optimizer.zero_grad(set_to_none=True)
            src,dst,t,msg = tensors(model,batch)
            logits, stage_logits = model.score(src,dst,msg)
            y = torch.as_tensor(batch['label'],dtype=torch.float32,device=model.device)
            sy = torch.as_tensor(batch['stage'],device=model.device)
            loss,binary_loss,stage_loss,stage_denominator = supervised_loss(
                logits,stage_logits,y,sy,positive_tensor,stage_weights,args.stage_weight)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            # PyG TGN updates memory before backward, then detaches after step.
            model.update(src,dst,t,msg)
            loss.backward()
            gradient_norm = nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
            gradient_max = max(gradient_max,float(gradient_norm))
            optimizer.step()
            model.memory.detach()
            total_loss += float(loss.detach())*len(src)
            binary_sum += float(binary_loss.detach())*len(src)
            stage_sum += float(stage_loss.detach()*stage_denominator)
            stage_denominator_sum += float(stage_denominator)
            total += len(src)
        validation = predict(model,args.data,'validation',args.eval_batch_size)
        threshold,feasible = choose_threshold(validation['label'],validation['probability'],args.min_precision,args.max_fpr,run['min_recall'])
        validation_metrics = metrics(validation['label'],validation['probability'],threshold)
        ap = validation_metrics['average_precision']
        score = ((int(feasible),validation_metrics['recall'] if feasible else 0.,ap,-validation_metrics['log_loss'])
                 if run['selection_metric']=='constrained_recall' else (ap,-validation_metrics['log_loss']))
        history.append(dict(epoch=epoch+1,train_loss=total_loss/total,train_binary_loss=binary_sum/total,
            train_stage_loss=stage_sum/stage_denominator_sum if stage_denominator_sum else None,
            max_gradient_norm_before_clipping=gradient_max,validation_average_precision=ap,
            validation_log_loss=validation_metrics['log_loss'],validation_precision=validation_metrics['precision'],
            validation_recall=validation_metrics['recall'],validation_f1=validation_metrics['f1'],
            validation_false_positive_rate=validation_metrics['false_positive_rate'],
            validation_threshold=threshold,validation_target_met=bool(feasible)))
        print(json.dumps(history[-1]),flush=True)
        if best is None or score > best:
            best,stale = score,0
            torch.save(model.checkpoint(dict(epoch=epoch+1,validation_average_precision=ap,selection_metric=run['selection_metric'],
                       dataset_sha256=run['dataset_sha256'],stage_training_counts=stage_counts.tolist())),output/'best.pt')
        else:
            stale += 1
        (output/'training-history.json').write_text(json.dumps(history,indent=2)+'\n')
        if stale >= args.patience:
            break
    model, checkpoint = TrainableTGN.restore(output/'best.pt',args.device)
    validation = predict(model,args.data,'validation',args.eval_batch_size)
    threshold, feasible = choose_threshold(validation['label'],validation['probability'],args.min_precision,args.max_fpr,run['min_recall'])
    calibration = predict(model,args.data,'calibration',args.eval_batch_size)
    benign = calibration['probability'][calibration['label']==0]
    if not len(benign):
        raise ValueError('Calibration split requires held-out benign examples')
    rank = math.ceil((len(benign)+1)*(1-args.alpha))
    conformal_q = float(np.sort(benign)[rank-1]) if rank<=len(benign) else 1.
    np.savez(output/'calibration.npz',benign_attack_probabilities=benign)
    operating = dict(threshold=threshold,validation_target_met=bool(feasible),min_precision=args.min_precision,
                     max_fpr=args.max_fpr,min_recall=run['min_recall'],batch_size=args.eval_batch_size,alpha=args.alpha,conformal_q=conformal_q,
                     calibration_benign_count=len(benign),checkpoint_sha256=sha256(output/'best.pt'),
                     calibration_sha256=sha256(output/'calibration.npz'),
                     conformal_comparison='score > conformal_q',
                     conformal_assumption='Exchangeability is required for a finite-sample coverage guarantee; temporal traffic may violate it.',
                     note='Threshold selected on validation only; test is evaluated separately. Calibration applies to raw model scores, not heuristic fusion.')
    (output/'operating-point.json').write_text(json.dumps(operating,indent=2)+'\n')
    (output/'training-history.json').write_text(json.dumps(history,indent=2)+'\n')
    (output/'validation.json').write_text(json.dumps(report_predictions(validation,threshold),indent=2)+'\n')
    print(json.dumps({'checkpoint':str(output/'best.pt'),'validation_target_met':bool(feasible),'threshold':threshold}),flush=True)


def evaluate(args):
    if args.threads < 1 or args.short_window < 1:
        raise ValueError('Positive threads and short-window required')
    torch.set_num_threads(args.threads)
    model, checkpoint = TrainableTGN.restore(args.checkpoint,args.device)
    operating = json.loads(Path(args.operating_point).read_text())
    if operating['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Operating point belongs to a different checkpoint')
    if checkpoint['dataset_sha256'] != sha256(Path(args.data)/'dataset.json'):
        raise ValueError('Evaluation dataset differs from the checkpoint manifest; reserve external captures as test groups before training')
    if not math.isfinite(operating['threshold']) or not 0 <= operating['threshold'] <= 1:
        raise ValueError('Operating-point threshold must be finite and in [0,1]')
    for target in ('min_precision','max_fpr','min_recall'):
        if target in operating and not 0 <= operating[target] <= 1:
            raise ValueError(f'Invalid operating-point {target}')
    if checkpoint['model_config']['num_nodes'] < read_metadata(args.data)['max_nodes']:
        raise ValueError('Evaluation capture exceeds checkpoint node capacity')
    output = Path(args.output)
    if output.exists():
        raise ValueError('Evaluation output already exists; choose a new report path')
    reports = {}
    support = set()
    for _, shard in shards(args.data, 'test'):
        support.update(shard['label'].tolist())
    if support != {0,1}:
        raise ValueError('Test requires both binary labels for meaningful evaluation')
    for size in map(int,args.batch_sizes.split(',')):
        if not 1<=size<=200 or size>model.config['history_capacity']:
            raise ValueError('Invalid evaluation batch size')
        for window in (0,args.short_window):
            predictions = predict(model,args.data,'test',size,window)
            reports[f'batch_{size}_reset_{window}'] = report_predictions(predictions,operating['threshold'])
    result = dict(checkpoint_sha256=sha256(args.checkpoint),dataset_sha256=sha256(Path(args.data)/'dataset.json'),
                  threshold=operating['threshold'],reports=reports,
                  note='Cold reset windows end on batch boundaries. Alternate batch modes require separate validation/calibration before deployment.')
    if 'min_recall' in operating:
        result['target_checks'] = {name:assess_protocol(r,operating['min_precision'],operating['min_recall'],operating['max_fpr']) for name,r in reports.items()}
        required = [f"batch_{operating['batch_size']}_reset_{w}" for w in (0,args.short_window)]
        passed = operating['validation_target_met'] and all(name in result['target_checks'] and result['target_checks'][name]['empirical_targets_met'] for name in required)
        result['binary_evidence_status'] = 'empirical_targets_met' if passed else 'insufficient_or_failed'
    else:
        result['binary_evidence_status'] = 'insufficient_or_failed'
    result['deployment_approved'] = False
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'report':str(output),'protocols':list(reports),'binary_evidence_status':result['binary_evidence_status']}))
    if getattr(args,'require_targets',False) and result['binary_evidence_status'] != 'empirical_targets_met':
        raise SystemExit(2)


def validate_train_args(args):
    weight = getattr(args,'positive_weight','auto')
    if weight != 'auto' and (not math.isfinite(float(weight)) or float(weight)<=0):
        raise ValueError('positive_weight must be auto or a finite positive number')
    if not 0 <= getattr(args,'min_recall',.95) <= 1:
        raise ValueError('min_recall must be in [0,1]')
    if getattr(args,'selection_metric','constrained_recall') not in ('constrained_recall','average_precision'):
        raise ValueError('Invalid selection_metric')
    for name in ('epochs','patience','node_capacity','memory_dim','embedding_dim',
                 'time_dim','direct_dim','history_capacity','threads'):
        if getattr(args, name) < 1:
            raise ValueError(f'{name} must be positive')
    if args.embedding_dim % 2:
        raise ValueError('embedding_dim must be even')
    for name in ('learning_rate','grad_clip','weight_decay','stage_weight'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (name in ('learning_rate','grad_clip') and value == 0):
            raise ValueError(f'Invalid {name}')
    if not 0 < args.alpha < 1 or not 0 <= args.min_precision <= 1 or not 0 <= args.max_fpr <= 1:
        raise ValueError('Invalid calibration/operating-point targets')
    if not 0 <= args.seed < 2**32:
        raise ValueError('seed must be in [0, 2**32)')
    if not 1 <= args.eval_batch_size <= min(200,args.history_capacity):
        raise ValueError('Evaluation batch size must fit history capacity and be at most 200')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest',required=True)
    prep.add_argument('--output',required=True)
    prep.add_argument('--shard-size',type=int,default=100000)
    fit = sub.add_parser('train')
    fit.add_argument('--data',required=True)
    fit.add_argument('--output',required=True)
    fit.add_argument('--epochs',type=int,default=30)
    fit.add_argument('--patience',type=int,default=5)
    fit.add_argument('--batch-sizes',default='1,32,200')
    fit.add_argument('--eval-batch-size',type=int,default=32)
    fit.add_argument('--node-capacity',type=int,default=34668)
    fit.add_argument('--memory-dim',type=int,default=128)
    fit.add_argument('--embedding-dim',type=int,default=128)
    fit.add_argument('--time-dim',type=int,default=32)
    fit.add_argument('--direct-dim',type=int,default=64)
    fit.add_argument('--history-capacity',type=int,default=100000)
    fit.add_argument('--time-transform',choices=['identity','signed_log1p'],default='identity')
    fit.add_argument('--learning-rate',type=float,default=.0003)
    fit.add_argument('--weight-decay',type=float,default=.0001)
    fit.add_argument('--stage-weight',type=float,default=.2)
    fit.add_argument('--grad-clip',type=float,default=1.)
    fit.add_argument('--seed',type=int,default=42)
    fit.add_argument('--min-precision',type=float,default=.9)
    fit.add_argument('--min-recall',type=float,default=.95)
    fit.add_argument('--positive-weight',default='auto',help='auto: sqrt(benign/attack), capped at 20; or a positive number (1 disables weighting)')
    fit.add_argument('--selection-metric',choices=['constrained_recall','average_precision'],default='constrained_recall')
    fit.add_argument('--max-fpr',type=float,default=.001)
    fit.add_argument('--alpha',type=float,default=.01)
    test = sub.add_parser('evaluate')
    test.add_argument('--data',required=True)
    test.add_argument('--checkpoint',required=True)
    test.add_argument('--operating-point',required=True)
    test.add_argument('--output',required=True)
    test.add_argument('--batch-sizes',default='1,32,200')
    test.add_argument('--short-window',type=int,default=1000)
    test.add_argument('--require-targets',action='store_true',help='Write the report, then exit 2 if validation or held-out serving-protocol targets are unmet or missing')
    for command in (fit,test):
        command.add_argument('--device',default='cpu')
        command.add_argument('--threads',type=int,default=2)
    args = parser.parse_args()
    if args.command=='prepare':
        prepare(args.manifest,args.output,args.shard_size)
    elif args.command=='train':
        if not 0<args.alpha<1 or not 0<=args.min_precision<=1 or not 0<=args.max_fpr<=1:
            parser.error('Invalid calibration/operating-point targets')
        if args.embedding_dim%2 or min(args.memory_dim,args.embedding_dim,args.time_dim,args.direct_dim,args.threads)<1:
            parser.error('Dimensions/threads must be positive and embedding dimension even')
        if not 1<=args.eval_batch_size<=min(200,args.history_capacity):
            parser.error('Evaluation batch size must fit history capacity and be at most 200')
        train(args)
    else:
        if args.short_window<1:
            parser.error('short-window must be positive')
        evaluate(args)
