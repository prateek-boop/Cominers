"""Audited conversion of the original CICIDS2017 TrafficLabelling export."""
import argparse, csv, hashlib, json, math, re, time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from ipaddress import ip_address
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.data import ALIASES, normalize
from model_runtime import FEATURE_NAMES

LABELS = ['BENIGN','FTP-Patator','SSH-Patator','DoS slowloris','DoS Slowhttptest','DoS Hulk','DoS GoldenEye','Heartbleed','Web Attack - Brute Force','Web Attack - XSS','Web Attack - Sql Injection','Infiltration','Bot','PortScan','DDoS']
DAYS = {'Monday':(3,'calibration'),'Tuesday':(4,'train'),'Wednesday':(5,'train'),'Thursday':(6,'validation'),'Friday':(7,'test')}

@lru_cache(maxsize=100000)
def capture_time(text, day):
    # Published capture is daytime. Original minute-resolution exports lose AM/PM.
    fmt = '%d/%m/%Y %H:%M:%S' if text.count(':') == 2 else '%d/%m/%Y %H:%M'
    value = datetime.strptime(text.strip(),fmt)
    if (value.year,value.month,value.day) != (2017,7,day):
        raise ValueError('timestamp_date_mismatch')
    if value.hour < 8:
        value = value.replace(hour=value.hour+12)
    if not 8 <= value.hour <= 18:
        raise ValueError('timestamp_outside_daytime_capture')
    # UTC is an explicit clock anchor, not a claim about the original capture timezone.
    return value.replace(tzinfo=timezone.utc).timestamp()

@lru_cache(maxsize=100000)
def valid_ip(value):
    return str(ip_address(value.strip()))

def convert(source, output):
    output.mkdir(parents=True,exist_ok=False)
    files=sorted(source.glob('*.csv'))
    if len(files)!=8: raise ValueError('Expected the eight original TrafficLabelling CSV files')
    report=dict(started=time.time(),source=str(source.resolve()),files=[],policy='Original source untouched; invalid rows quarantined, no numeric imputation. Day-first dates; hours 0-7 interpreted as afternoon using published daytime schedule; UTC clock anchor only, original timezone unverified. Binary research candidate; stage labels masked.',schedule_source='https://www.unb.ca/cic/datasets/ids-2017.html')
    headers=['src_ip','dst_ip','timestamp','label',*FEATURE_NAMES]
    handles={day:(output/(day.lower()+'.csv')).open('w',newline='') for day in DAYS}
    writers={day:csv.writer(f) for day,f in handles.items()}
    for w in writers.values(): w.writerow(headers)
    try:
        with (output/'quarantine.jsonl').open('w') as rejected:
            for path in files:
                day=next(d for d in DAYS if path.name.lower().startswith(d.lower()))
                stats=dict(file=path.name,rows=0,accepted=0,reasons=Counter(),labels=Counter())
                digest=hashlib.sha256()
                with path.open('rb') as raw:
                    for chunk in iter(lambda:raw.read(1024*1024),b''): digest.update(chunk)
                stats['sha256']=digest.hexdigest()
                # Windows-1252 preserves the original en dash in Web Attack labels.
                with path.open(encoding='cp1252',newline='') as stream:
                    reader=csv.reader(stream); fields=next(reader)
                    indexed={}
                    for i,name in enumerate(fields): indexed.setdefault(normalize(name),[]).append(i)
                    aliases=dict(zip(FEATURE_NAMES,ALIASES))
                    aliases.update(src_ip=['Source IP'],dst_ip=['Destination IP'],timestamp=['Timestamp'],label=['Label'])
                    positions={}
                    for key,names in aliases.items():
                        found={i for name in names for i in indexed.get(normalize(name),[])}
                        if len(found)!=1: raise ValueError(f'Ambiguous required column {key}')
                        positions[key]=found.pop()
                    for line,row in enumerate(reader,2):
                        stats['rows']+=1
                        try:
                            if len(row)!=len(fields): raise ValueError('row_width')
                            values=[float(row[positions[k]]) for k in FEATURE_NAMES]
                            if any(not math.isfinite(v) or v<0 for v in values): raise ValueError('invalid_numeric')
                            if any(math.log1p(v)>100 for v in values): raise ValueError('numeric_range')
                            src,dst=(valid_ip(row[positions[k]]) for k in ('src_ip','dst_ip'))
                            timestamp=capture_time(row[positions['timestamp']],DAYS[day][0])
                            label=row[positions['label']].strip().replace('\u2013','-').replace('\u2014','-')
                            label=re.sub(r'\s+',' ',label)
                            if label not in LABELS: raise RuntimeError(f'Unknown label {label!r} at {path}:{line}')
                            writers[day].writerow([src,dst,format(timestamp,'.6f'),label,*values])
                            stats['accepted']+=1;stats['labels'][label]+=1
                        except ValueError as exc:
                            reason=str(exc);stats['reasons'][reason]+=1
                            rejected.write(json.dumps(dict(file=path.name,line=line,reason=reason,row=row))+'\n')
                report['files'].append(stats)
                (output/'conversion-report.json').write_text(json.dumps(report,indent=2))
                print(json.dumps(stats),flush=True)
    finally:
        for f in handles.values(): f.close()
    manifest=dict(labels={l:dict(attack=int(l!='BENIGN'),stage=None) for l in LABELS},
        sources=[dict(group='cic2017-'+day.lower(),split=split,path=day.lower()+'.csv',timestamp_kind='start') for day,(_,split) in DAYS.items()],
        note=report['policy'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report['elapsed_seconds']=time.time()-report['started']
    (output/'conversion-report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();convert(a.source,a.output)
