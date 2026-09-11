"""Read-only schema and bounded-row audit; does not establish full dataset integrity."""
import ast, csv, json, math, re
from pathlib import Path
from collections import Counter
from ipaddress import ip_address
from datetime import datetime, timezone
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'reports/audit/dataset-readiness'
def literal(path, name):
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
def normalize(s): return re.sub('[^a-z0-9]', '', s.lower())
features = literal(ROOT/'model_runtime.py','FEATURE_NAMES')
aliases = literal(ROOT/'training/data.py','ALIASES')
required = dict(zip(features, aliases))
required.update(src_ip=['src_ip','Source IP','Src IP'],dst_ip=['dst_ip','Destination IP','Dst IP'],timestamp=['Timestamp'],label=['Label'])
results=[]
for folder in ('TrafficLabelling ', 'MachineLearningCVE', 'cic-ids2018-processed'):
    for path in sorted((ROOT/folder).glob('*')):
        if not path.is_file(): continue
        before=path.stat()
        with path.open('rb') as f:
            f.seek(max(0,before.st_size-16384))
            tail=f.read()
        with path.open(newline='',encoding='utf-8-sig',errors='replace') as f:
            reader=csv.reader(f); headers=next(reader)
            indexed={}
            for i,h in enumerate(headers): indexed.setdefault(normalize(h),[]).append(i)
            selected={k:sorted({i for a in names for i in indexed.get(normalize(a),[])}) for k,names in required.items()}
            missing=[k for k,v in selected.items() if not v]
            duplicates=[headers[v[0]].strip() for v in indexed.values() if len(v)>1]
            labels=Counter(); errors=Counter(); rows=0; first_timestamp=None
            for row in reader:
                rows+=1
                if len(row)!=len(headers): errors['wrong_column_count']+=1
                else:
                    if any('\ufffd' in s for s in row): errors['utf8_replacement_in_row']+=1
                    if selected['label']:
                        label=row[selected['label'][0]].strip(); labels[label]+=1
                        if not label: errors['empty_label']+=1
                    for key in features:
                        if not selected[key]: continue
                        try:
                            value=float(row[selected[key][0]])
                            if not math.isfinite(value) or value<0: errors['invalid_required_numeric_cell']+=1
                        except ValueError: errors['invalid_required_numeric_cell']+=1
                    for key in ('src_ip','dst_ip'):
                        if selected[key]:
                            try: ip_address(row[selected[key][0]].strip())
                            except ValueError: errors['invalid_ip_cell']+=1
                    if selected['timestamp'] and first_timestamp is None: first_timestamp=row[selected['timestamp'][0]]
                if rows>=5000: break
        after=path.stat()
        results.append(dict(file=str(path.relative_to(ROOT)),bytes=after.st_size,columns=len(headers),missing_required=missing,
            duplicate_header_names=duplicates,sampled_rows=rows,sample_labels=dict(labels),sample_errors=dict(errors),
            first_timestamp=first_timestamp,non_csv_suffix=path.suffix!='.csv',ends_with_newline=tail.endswith(b'\n'),
            final_record_field_count=len(next(csv.reader([tail.decode('utf-8',errors='replace').splitlines()[-1]]))),
            changed_during_check=before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns))
manifest=json.loads((ROOT/'examples/training/manifest.json').read_text())
result=dict(checked_at=datetime.now(timezone.utc).isoformat(),scope='Headers, file sizes/tails and first 5000 rows per file; no full-row, checksum or duplicate-content verification',
    files=results,manifest_paths=[dict(split=s['split'],exists=((ROOT/'examples/training')/s['path']).exists()) for s in manifest['sources']])
(OUT/'inventory.json').write_text(json.dumps(result,indent=2)+'\n')
for folder in ('TrafficLabelling ', 'MachineLearningCVE','cic-ids2018-processed'):
    rows=[r for r in results if r['file'].startswith(folder+'/')]
    print(json.dumps(dict(folder=folder,files=len(rows),bytes=sum(r['bytes'] for r in rows),sampled_rows=sum(r['sampled_rows'] for r in rows),
        missing_required=sorted({k for r in rows for k in r['missing_required']}),duplicate_headers=sorted({k for r in rows for k in r['duplicate_header_names']}),
        sample_errors=dict(sum((Counter(r['sample_errors']) for r in rows),Counter())),temporary_files=[r['file'] for r in rows if r['non_csv_suffix']],
        incomplete_final_records=[r['file'] for r in rows if r['final_record_field_count']!=r['columns']])))
