# Exploratory speculative decoding research

The subsequent GPU experiment is in `gpu_experiment/`. Its `RESULTS.txt`
documents the restricted online edit-rule implementation, 160 A100 trials,
negative controls, code-quality checks and throughput limits. The experiments
below describe the earlier proposal stage and remain as historical evidence.

This directory contains a November 24, 2026 research pilot, not a vLLM integration
or a completed paper. The proposed direction is to infer repeated edit rules
from already committed target output and use a CPU to apply those rules when
constructing speculative continuations. The target still verifies every token.

## Evidence and limits

`pilot.py` generated eight ordinary chat rewrites with Qwen2.5-Coder-0.5B-Instruct.
The model ignored most renaming requests. Those failures remain in `traces.json`
and `pilot_results.json`; they are not removed or counted as successful editing.

`completion_pilot.py` separately supplied a renamed function header as an
assistant prefix and generated a body. These are constructed completion cases,
not ordinary chat, and the supplied header is excluded from generated-token
counts. The target sometimes generated poor code. Matching its output is not a
claim to improve code quality.

`verify_benchmark.py` ran an actual greedy CPU draft/verify loop with KV-cache
rollback, randomizing 72 runs: 8 cases x 3 methods x 3 repetitions. Qwen model
revision is recorded in `model/source.json`; CPU is Intel Xeon Silver 4210R,
4 PyTorch threads, FP32, SDPA, Transformers 4.57.1, PyTorch 2.11.0+cu130.
No GPU was used. Median total times across the eight cases were:

| Method | Seconds | Target decode calls |
| --- | ---: | ---: |
| Ordinary greedy decoding | 15.635 | 138 |
| Exact n-gram speculation | 16.364 | 95 |
| Parameterized matcher with exact fallback | 13.809 | 75 |

Times include prefill, proposal generation, verification, and rollback. They
exclude loading, warmup, initial tokenization and index construction. The
baseline is a Python longest-match scan with earliest-occurrence tie breaking,
2–8 BPE-token windows, and at most 8 draft tokens, not compiled vLLM KMP.
The structural index uses 3–24 Python lexical units, so this is not a controlled
comparison of matching algorithms at equal match lengths. All 72 outputs were
token-identical within each case. Timing gains are exploratory; no confidence
interval, serving load, long-context test, NVIDIA CPU measurement, stochastic
sampling check, or production-runtime speedup is established.

`corpus_pilot.py` applied the same prefix-conditioned construction to the first
16 eligible HumanEval source functions (selection and source digest recorded).
Reference solutions were visible in the prompts. This is NOT an official
HumanEval accuracy benchmark. On 630 target-generated tokens, oracle replay
needed 372 exact-matching steps, 452 structural-only steps, or 349 hybrid steps.
The hybrid was better on 6 cases, tied on 8 and worse on 2. These are offline
step counts, not measured serving speedups. Outputs capped at 128 new tokens
remain included even if incomplete.

The critical negative control is `heuristic_control.py`: rename source
parameters by position in the observed new header, then use ordinary matching.
It needed only 56 steps on the eight constructed examples and 282 steps on the
16 corpus examples. This outperformed the structural hybrid's 75 and 349 steps.
Therefore, the implemented matcher is NOT established as a contribution beyond
a straightforward renaming heuristic. Do not present its CPU timing improvement
as evidence for the proposed, broader edit-rule method.

`edit_rule_proof.py` implements a separate joint anti-unification mechanism:
two observed call rewrites share argument holes between source and target
templates. It then proposes an unseen argument with an inserted keyword and
changed callee. Six assertions cover a positive example, unrelated calls,
changed literal preconditions, repeated-argument consistency and an
underdetermined target. These are constructed symbolic checks only. Automatic
source/output alignment, collection of actual accepted edit pairs, concrete
syntax preservation, runtime integration, and model-level timing for this
broader proposal are not implemented.

## Reproduce on this host

The model is public, Apache-2.0, with immutable revision in `model/source.json`.
The pilot reuses the existing read-only dependency directory from aria-style-lab.

```sh
/var/tmp/work-mshanthappa/finstak-runtime/bin/python pilot.py
/var/tmp/work-mshanthappa/finstak-runtime/bin/python completion_pilot.py
/var/tmp/work-mshanthappa/finstak-runtime/bin/python verify_benchmark.py
/var/tmp/work-mshanthappa/finstak-runtime/bin/python corpus_pilot.py
/var/tmp/work-mshanthappa/finstak-runtime/bin/python heuristic_control.py
/var/tmp/work-mshanthappa/finstak-runtime/bin/python test_invariants.py
python edit_rule_proof.py
```

Run from this directory. Reruns overwrite the corresponding result files.
For another host, use a separate environment with the recorded dependency
versions, adapt the read-only dependency path, and download the exact recorded
model revision. The broader paper proposal is saved separately in a private Page.
