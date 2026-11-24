# Measured snapshots and portable entry points

The recorded results and all `source_snapshot/` directories are copied unchanged from the original experiment. Their original names and internal method labels are retained. Their protocol hashes remain authoritative for the reported run.

The working copy of `research/gpu_experiment/benchmark.py` has packaging changes: remove an absolute dependency path and forced offline environment variable; accept `--model-path`; require a new `--output-dir`; reject non-positive repeat counts; and detect the current CPU rather than recording the original machine's CPU name. The decoding and proposer algorithms are unchanged. The source snapshot contains the original runner.

The top-level GPU README points to portable instructions. Historical scripts and pilot manifests are archival and can include paths from the original host. They contain no credentials or model weights. `ARTIFACT_MANIFEST.json` covers the current package; older manifests cover the earlier archives and frozen runs.

MorphDraft was previously named RuleLoom in the draft manuscript. The paper, figures, and current report use MorphDraft. Original experiment archives predate this naming change and are preserved as provenance.
