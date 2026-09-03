import hashlib
import json

def hash_record(record: dict) -> bytes:
    serialized = json.dumps(record, sort_keys=True).encode('utf-8')
    return hashlib.sha256(serialized).digest()
