"""Read-only historical ledger and saved evaluation consistency audit.

Writes derived reports and valid ledger branches under reports/audit. Never
rewrites the operational ledger or claims that saved metrics are a fresh test.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/audit'


def merkle(leaves):
    level = list(leaves)
    if not level:
        return None
    while len(level) > 1:
        level = [hashlib.sha256((level[i] + level[min(i+1, len(level)-1)]).encode()).hexdigest()
                 for i in range(0, len(level), 2)]
    return level[0]


def audit_ledger():
    path = ROOT / 'ledger/ledger.jsonl'
    raw = path.read_bytes()
    blocks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    nodes = {b['block_hash']: b for b in blocks}
    invalid = []
    for pos, block in enumerate(blocks):
        header = {k:block[k] for k in ('block_index','timestamp','prev_block_hash','merkle_root','leaf_count')}
        actual_hash = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if actual_hash != block['block_hash'] or merkle(block['leaf_hashes']) != block['merkle_root'] or len(block['leaf_hashes']) != block['leaf_count']:
            invalid.append(pos)
    parents = {b['prev_block_hash'] for b in blocks}
    tips = [b for b in blocks if b['block_hash'] not in parents]
    branches = []
    if not invalid:
        for tip in tips:
            branch, seen = [], set()
            node = tip
            while node:
                if node['block_hash'] in seen:
                    raise ValueError('Ledger cycle')
                seen.add(node['block_hash'])
                branch.append(node)
                if node['prev_block_hash'] == '0' * 64:
                    break
                node = nodes.get(node['prev_block_hash'])
                if node is None:
                    raise ValueError('Missing parent block')
            branch.reverse()
            valid = all(b['block_index'] == i for i,b in enumerate(branch))
            name = OUT / 'ledger-branches' / (tip['block_hash'][:16] + '.jsonl')
            if valid:
                name.parent.mkdir(parents=True, exist_ok=True)
                name.write_text(''.join(json.dumps(b) + '\n' for b in branch))
            branches.append({'tip':tip['block_hash'], 'blocks':len(branch), 'valid':valid,
                             'derived_path':str(name.relative_to(ROOT)) if valid else None})
    # Rebuild only a derived view; changed headers are explicitly mapped back
    # to their original evidence hashes. This does not authenticate event order.
    recovered = []
    mapping = []
    previous = '0' * 64
    if not invalid:
        for index, original in enumerate(blocks):
            block = dict(original, block_index=index, prev_block_hash=previous)
            header = {k:block[k] for k in ('block_index','timestamp','prev_block_hash','merkle_root','leaf_count')}
            block['block_hash'] = hashlib.sha256(json.dumps(header,sort_keys=True,separators=(',',':')).encode()).hexdigest()
            recovered.append(block)
            mapping.append({'position':index, 'original_hash':original['block_hash'], 'derived_hash':block['block_hash']})
            previous = block['block_hash']
        (OUT/'recovered-ledger-view.jsonl').write_text(''.join(json.dumps(b)+'\n' for b in recovered))
        (OUT/'recovered-ledger-provenance.json').write_text(json.dumps(mapping,indent=2)+'\n')
    recovered_valid = bool(recovered)
    for i, block in enumerate(recovered):
        recovered_valid &= block['block_index'] == i
        recovered_valid &= block['prev_block_hash'] == ('0' * 64 if i == 0 else recovered[i-1]['block_hash'])
        recovered_valid &= merkle(block['leaf_hashes']) == block['merkle_root']
        header = {k:block[k] for k in ('block_index','timestamp','prev_block_hash','merkle_root','leaf_count')}
        recovered_valid &= block['block_hash'] == hashlib.sha256(json.dumps(header,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    report = {'source':str(path.relative_to(ROOT)), 'original_sha256':hashlib.sha256(raw).hexdigest(),
              'records':len(blocks), 'invalid_record_hashes_or_merkle_roots':invalid,
              'forks':len(tips), 'derived_branches':branches,
              'recovered_view_integrity_valid':recovered_valid,
              'recovered_view':'reports/audit/recovered-ledger-view.jsonl' if recovered else None,
              'recovered_view_preserves_all_leaf_hashes':[b['leaf_hashes'] for b in recovered]==[b['leaf_hashes'] for b in blocks],
              'original_unchanged':path.read_bytes() == raw,
              'interpretation':'Derived branches preserve original blocks. No branch is automatically chosen as authoritative.'}
    (OUT/'ledger-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


def audit_metrics():
    checkpoint_hash = hashlib.sha256((ROOT/'checkpoints/tgn_best.pt').read_bytes()).hexdigest()
    reports = []
    for path in sorted((ROOT/'reports').glob('model_evaluation*.json')):
        data = json.loads(path.read_text())
        metrics = data['metrics']
        cm = metrics['confusion_matrix']
        tn, fp, fn, tp = (cm[k] for k in ('tn','fp','fn','tp'))
        total = sum(cm.values())
        expected = {'events':total, 'attacks':tp+fn, 'benign':tn+fp,
                    'accuracy':(tp+tn)/total, 'precision':tp/(tp+fp), 'recall':tp/(tp+fn),
                    'f1':2*tp/(2*tp+fp+fn), 'false_positive_rate':fp/(fp+tn)}
        inconsistent = [k for k,v in expected.items() if not math.isclose(metrics[k],v,abs_tol=1e-12)]
        chunk_counts = {k:sum(c['confusion_matrix'][k] for c in data['per_chunk']) for k in cm}
        protocol = data['protocol']
        overlap = sorted(set(protocol['train_files']) & set(protocol['validation_files']))
        reports.append({'file':path.name, 'matches_retained_checkpoint':data['checkpoint_sha256']==checkpoint_hash,
                        'inconsistent_metrics':inconsistent, 'chunk_counts_match':chunk_counts==cm,
                        'train_validation_file_overlap':overlap, 'events':total,
                        'precision':expected['precision'], 'recall':expected['recall'], 'f1':expected['f1'],
                        'batch_size':protocol['batch_size'], 'stage_macro_f1':metrics['stage_macro_f1']})
    report = {'checkpoint_sha256':checkpoint_hash, 'reports':reports,
              'fresh_accuracy_evaluation':False, 'reason':'Labeled dataset and source evaluator are absent.',
              'auc_and_average_precision_recomputed':False,
              'limitation':'Aggregate count consistency does not establish dataset provenance or independent accuracy.'}
    (OUT/'model-evidence-verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'ledger':audit_ledger(), 'model':audit_metrics()},indent=2))
