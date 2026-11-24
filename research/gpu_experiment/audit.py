"""Independently verify result accounting and the saved greedy execution traces."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent


def audit(path):
    protocol=json.loads((path/'protocol.json').read_text())
    results=json.loads((path/'results.json').read_text())
    rows=results['records']
    references=results['references']
    expected=protocol['repetitions']*len(protocol['tasks'])*len(protocol['methods'])
    assert len(rows)==expected
    keys={(r['repeat'],r['case'],r['method']) for r in rows}
    assert len(keys)==expected
    traces=steps=0
    for row in rows:
        ref=references[row['case']]
        assert row['ids']==ref
        assert row['tokens']==len(row['ids'])
        assert row['matches_reference']
        assert row['tokens_per_second']==row['tokens']/row['seconds']
        if row['repeat']!=0:continue
        traces+=1
        assert len(row['step_trace'])==row['steps']
        prefix=[]
        for step in row['step_trace']:
            steps+=1
            assert step['output_position']==len(prefix)
            assert step['accepted']<=len(step['draft'])
            assert step['committed'][:step['accepted']]==step['draft'][:step['accepted']]
            prefix.extend(step['committed'])
            assert prefix==ref[:len(prefix)]
        assert prefix==ref
    for filename,digest in protocol['source_hashes'].items():
        assert hashlib.sha256((path/'source_snapshot'/filename).read_bytes()).hexdigest()==digest
    independent=json.loads((path/'independent_checks.json').read_text())
    assert all(r['matches_generate'] for r in independent)
    report={'passed':True,'rows':len(rows),'distinct_case_method_repeat_keys':len(keys),
            'traces_checked':traces,'verification_steps_checked':steps,
            'all_outputs_token_identical':True,'independent_generate_checks':independent,
            'source_snapshot_hashes_match':True,
            'scope':'Accounting and saved trace audit; does not establish real-world generalization or sampling correctness'}
    (path/'audit.json').write_text(json.dumps(report,indent=2))
    print(path.name,json.dumps(report))


if __name__=='__main__':
    for name in ['small_smoke','7b_smoke','small_measured','7b_measured']:
        p=HERE/name
        if (p/'completed.json').exists():audit(p)
