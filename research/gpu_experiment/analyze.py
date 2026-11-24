"""Audit equality, requested edits and paired wall-clock throughput."""
import ast
import copy
import csv
import hashlib
import json
from pathlib import Path
import random
import statistics

HERE=Path(__file__).resolve().parent
METHODS=['autoregressive','ngram','heuristic','rules']


def check_edit(task, text):
    expected=ast.parse(task['source'])
    kind=task['kind']
    for i,node in enumerate(expected.body[0].body):
        if not isinstance(node,ast.Assign):continue
        call=node.value
        if kind=='unchanged' or (kind=='one_edit' and i!=0):continue
        if kind=='reorder':
            call.func=ast.parse('client.send',mode='eval').body
            call.args=call.args[::-1]
            call.keywords.append(ast.keyword(arg='secure',value=ast.Constant(True)))
        elif kind=='mixed' and i%2:
            call.func=ast.parse('client.store',mode='eval').body
            call.keywords.append(ast.keyword(arg='durable',value=ast.Constant(True)))
        else:
            call.func=ast.parse('client.fetch',mode='eval').body
            call.keywords.append(ast.keyword(arg='timeout',value=ast.Constant(10)))
    clean=text
    if clean.startswith('```python\n'):clean=clean[len('```python\n'):]
    clean=clean.split('```')[0].strip()
    try:actual=ast.parse(clean)
    except SyntaxError:return {'parses':False,'matches_requested_AST':False}
    strict=ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)
    # All constructed keyword values are constants. Compare a second time
    # ignoring their syntactic order, without hiding missing/changed values.
    for tree in [expected,actual]:
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and all(k.arg is not None and isinstance(k.value,ast.Constant) for k in node.keywords):
                node.keywords.sort(key=lambda k:k.arg)
    return {'parses':True,'matches_requested_AST':strict,
            'matches_requested_ignoring_keyword_order':ast.dump(expected,include_attributes=False)==ast.dump(actual,include_attributes=False)}


def percentile(values, q):
    return sorted(values)[round((len(values)-1)*q)]


def analyze(path):
    protocol=json.loads((path/'protocol.json').read_text())
    raw=json.loads((path/'results.json').read_text())
    records=raw['records']
    complete=(path/'completed.json').exists()
    output_texts=json.loads((path/'reference_outputs.json').read_text()) if complete else {}
    task_by_name={t['name']:t for t in protocol['tasks']}
    quality={name:check_edit(task_by_name[name],text) for name,text in output_texts.items()}
    cases={}
    for name,task in task_by_name.items():
        cases[name]={'kind':task['kind'],'methods':{}}
        for method in METHODS:
            rs=[r for r in records if r['case']==name and r['method']==method]
            if not rs:continue
            cases[name]['methods'][method]={
                'median_seconds':statistics.median(r['seconds'] for r in rs),
                'median_tokens_per_second':statistics.median(r['tokens_per_second'] for r in rs),
                'tokens':rs[0]['tokens'],'steps':rs[0]['steps'],
                'median_proposal_seconds':statistics.median(r['proposal_seconds'] for r in rs),
                'median_setup_seconds':statistics.median(r['setup_seconds'] for r in rs),
                'median_prefill_seconds':statistics.median(r['prefill_seconds'] for r in rs),
                'all_outputs_match':all(r['matches_reference'] for r in rs),
                'first_rule_after_tokens':rs[0].get('proposer',{}).get('first_rule_after_tokens'),
                'rules':rs[0].get('proposer',{}).get('rules')}
    aggregate={}
    for method in METHODS:
        rs=[r for r in records if r['method']==method]
        if not rs:continue
        aggregate[method]={'pooled_tokens_per_second':sum(r['tokens'] for r in rs)/sum(r['seconds'] for r in rs),
            'total_seconds':sum(r['seconds'] for r in rs),'tokens':sum(r['tokens'] for r in rs),
            'all_outputs_match':all(r['matches_reference'] for r in rs),'runs':len(rs)}
    ratios={}
    if complete:
        names=list(cases)
        for other in ['autoregressive','ngram','heuristic']:
            rng=random.Random(2401)
            # Descriptive task-bootstrap for this small constructed workload;
            # it is not a generalization interval for real-world code edits.
            samples=[]
            for _ in range(10000):
                selected=[rng.choice(names) for i in names]
                a=sum(cases[n]['methods'][other]['median_seconds'] for n in selected)
                b=sum(cases[n]['methods']['rules']['median_seconds'] for n in selected)
                samples.append(a/b)
            point=sum(cases[n]['methods'][other]['median_seconds'] for n in names)/sum(cases[n]['methods']['rules']['median_seconds'] for n in names)
            ratios[other]={'paired_median_latency_speedup':point,
                'descriptive_task_bootstrap_95':[percentile(samples,.025),percentile(samples,.975)]}
    report={'complete':complete,'protocol_scope':protocol['scope'],'gpu':protocol['gpu'],
        'model':protocol['model'],'dtype':protocol['dtype'],'records':len(records),
        'all_outputs_match':all(r['matches_reference'] for r in records),
        'methods':aggregate,'cases':cases,'rules_vs':ratios,'requested_edit_audit':quality,
        'limitations':['constructed tasks with known source assignment identifiers',
            'batch size one and concurrency one; no serving-capacity result',
            'custom Transformers loop, not compiled vLLM baselines',
            'rule induction and all source rewriting charged to CPU proposal time',
            'no future target tokens or expected edited source are given to proposers',
            'quality audit does not filter timing rows']}
    (path/'summary.json').write_text(json.dumps(report,indent=2))
    with (path/'summary.csv').open('w') as f:
        fields=['case','kind','method','tokens','steps','median_seconds','median_tokens_per_second','all_outputs_match']
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        for name,case in cases.items():
            for method,row in case['methods'].items():
                w.writerow({'case':name,'kind':case['kind'],'method':method,**{k:row[k] for k in fields if k in row}})
    print(path.name,json.dumps({'complete':complete,'all_outputs_match':report['all_outputs_match'],
        'tps':{m:round(v['pooled_tokens_per_second'],2) for m,v in aggregate.items()},
        'quality':quality,'rules_vs':ratios}))
    return report


if __name__=='__main__':
    for name in ['small_smoke','7b_smoke','small_measured','7b_measured']:
        path=HERE/name
        if (path/'results.json').exists():analyze(path)
