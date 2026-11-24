# MorphDraft GPU verification with online CPU edit rules

This experiment measures a restricted implementation of the research proposal:
a CPU learns symbolic call transformations from completed target-generated
statements, builds a rewritten-source index, and proposes continuations for a
GPU target model to verify. It is a constructed mechanism study, not a vLLM
integration or a benchmark of production serving capacity.

## Methods

All four methods share `benchmark.py`'s greedy verifier, CUDA model, cache
handling and request timings:

1. `autoregressive`: no speculative proposals.
2. `ngram`: indexed exact matching over the full model prompt and committed
   output; longest match of 2–8 target tokens, earliest occurrence tie breaking,
   and at most 8 proposed tokens.
3. `heuristic`: a strong simple control. From one completed edit, copy the new
   callee and its literal keyword arguments to later source calls, retaining
   their positional arguments. It can insert, remove and replace fixed keyword
   arguments, but cannot infer positional-argument reordering.
4. `rules`: jointly generalize two completed source/output assignment pairs
   from the same original call family. Shared holes preserve relations between
   the before/after patterns, including argument reordering. Instantiate the
   resulting rule on source assignments and index the rewritten source. Fall
   back to exact matching when no rewritten continuation is available.

Rules see the known source and already committed output. They do not receive
expected edited source, model reference completions, or future output tokens.
The source/output alignment is deliberately restricted: matching unique
assignment-variable names in straight-line functions. Source formatting is
normalized through Python ASTs. These restrictions are not a general solution
to arbitrary code editing.

Completed observations require a newline boundary. A partial statement cannot
train a rule. The rule learner validates a generalized rule against past
matching observations and retires conflicting rules. No generated code or API
call is executed.

## Workload and measurement

`tasks.py` fixes eight tasks before the full measurement: repeated API migration,
positional-argument reordering, unchanged code, a single edited call, and two
interleaved API migrations. They contain 8–16 calls each. The function header is
supplied as an assistant prefix; all body generation remains the target's work.
The proposers are re-created for every request, so rules do not transfer across
trials or tasks.

The full experiment is `7b_measured`: Qwen2.5-Coder-7B-Instruct, BF16, one NVIDIA
A100 80 GB PCIe GPU, one CPU PyTorch thread, batch size one and concurrency one.
Five repetitions of eight tasks and four methods give 160 randomized requests.
Model revisions, hashes, source snapshots and exact timing scope are in its
`protocol.json`.

Request time includes tokenization, index construction, prefill, CPU learning,
CPU proposal generation, GPU verification, synchronization and KV rollback.
It excludes model loading, warmup, networking and queueing. All target steps use
the same forward-pass implementation. Tokens/s includes the generated EOS token
and excludes the supplied assistant prefix. No stochastic-sampling claim is made.

Small-model and 7B smoke tests use one repetition of four cases. The heuristic
was strengthened after those tests, before the full measurement; their original
source snapshots are preserved. Smoke and final measurements must not be pooled.

## Results and checks

The final measured result is recorded in `7b_measured/summary.json` and
`summary.csv`. Raw token IDs, request timings, component timings, and first-
repetition per-step traces are in `results.json`. `reference_outputs.json`
contains the ordinary target model's generated programs.

`analyze.py` separately checks compliance with the requested edits. It does not
filter timing results when the target model makes an editing mistake. Matching
the target output establishes preservation of that output on these requests,
not that the target's program is correct.

`audit.py` verifies all case/method/repetition keys, exact output equality,
throughput arithmetic, saved step traces, source snapshot hashes and independent
Transformers.generate checks. `test_proposers.py` tests indexed lookup against an
exhaustive matcher, completed-observation requirements, reordering, unchanged-
code behavior, contradictory feedback and the strengthened heuristic control.

An observed batch-one tokens/s improvement does not establish a gain under
continuous batching, high concurrency, long real-world files, or a compiled
vLLM/GPU suffix baseline. The fixed synthetic workload favors repeated edits;
real edit prevalence and arbitrary source alignment remain open questions.

## Portable reproduction

See the repository [README](../../README.md) for environment installation, pinned checkpoint download, safe output-directory selection, and audit commands. The measured source snapshots remain unchanged; the working benchmark has the packaging changes documented in [PORTABILITY.md](../../PORTABILITY.md).
