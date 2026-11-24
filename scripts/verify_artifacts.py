"""Verify package digests, final-run snapshot hashes, and pooled accounting."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'ARTIFACT_MANIFEST.json').read_text())
    for name,digest in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    run=ROOT/'research/gpu_experiment/7b_measured'
    protocol=json.loads((run/'protocol.json').read_text())
    for name,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((run/'source_snapshot'/name).read_bytes()).hexdigest()==digest,name
    rows=json.loads((run/'results.json').read_text())['records']
    summary=json.loads((run/'summary.json').read_text())
    assert len(rows)==160 and all(r['matches_reference'] for r in rows)
    for method,expected in summary['methods'].items():
        selected=[r for r in rows if r['method']==method]
        actual=sum(r['tokens'] for r in selected)/sum(r['seconds'] for r in selected)
        assert abs(actual-expected['pooled_tokens_per_second'])<1e-8,method
    print(f"Verified {len(manifest['files'])} artifact digests, measured source hashes, and {len(rows)} request records.")

if __name__=='__main__':main()
