"""Descriptive family-level workload sensitivity; not a generalization guarantee."""
import json
import random
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FAMILIES={'migration':['migrate8','migrate16'],'reorder':['reorder8','reorder16'],
          'unchanged':['unchanged8','unchanged16'],'one_edit':['one_edit8'],'mixed':['mixed12']}

def quantile(values,q):
    # Use an order statistic, matching the earlier exploratory analysis.
    return sorted(values)[round((len(values)-1)*q)]

def main():
    summary=json.loads((ROOT/'research/gpu_experiment/7b_measured/summary.json').read_text())
    cases=summary['cases']
    def ratio(names,baseline):
        return sum(cases[n]['methods'][baseline]['median_seconds'] for n in names)/sum(cases[n]['methods']['rules']['median_seconds'] for n in names)
    report={}
    for baseline in ['ngram','heuristic']:
        rng=random.Random(261124)
        samples=[]
        for _ in range(20000):
            names=[n for fam in rng.choices(list(FAMILIES),k=len(FAMILIES)) for n in FAMILIES[fam]]
            samples.append(ratio(names,baseline))
        report[baseline]={'paired_median_speedup':ratio(list(cases),baseline),
                          'descriptive_95':[quantile(samples,.025),quantile(samples,.975)],
                          'resamples':20000,'seed':261124,'cluster_unit':'constructed task family'}
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
