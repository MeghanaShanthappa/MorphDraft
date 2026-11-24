"""Exploratory CPU pilot. Not a vLLM implementation or a publication benchmark.

Binding metadata comes only from the supplied source and generated prefix.
The held-out suffix is consulted exclusively by replay(), never by a proposer.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, "/var/tmp/work-mshanthappa/aria-style-lab/deps")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import ast
import hashlib
import io
import json
import keyword
import platform
import random
import re
import statistics
import time
import tokenize
from collections import defaultdict


CASES = [
    ("mean", "def mean(values):\n    total = sum(values)\n    return total / len(values)\n", {"values": "observations"}),
    ("clamp", "def clamp(value, lower, upper):\n    return max(lower, min(value, upper))\n", {"value": "measurement", "lower": "minimum", "upper": "maximum"}),
    ("normalize", "def normalize(values):\n    total = sum(values)\n    return [value / total for value in values]\n", {"values": "measurements"}),
    ("distance", "def distance(x, y):\n    return ((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) ** 0.5\n", {"x": "first_point", "y": "second_point"}),
    ("weighted", "def weighted(values, weights):\n    return sum(value * weight for value, weight in zip(values, weights)) / sum(weights)\n", {"values": "observations", "weights": "importance"}),
    ("count", "def count(items, target):\n    return sum(1 for item in items if item == target)\n", {"items": "records", "target": "needle"}),
    ("unchanged", "def reverse(items):\n    return list(reversed(items))\n", {}),
    ("logic_change", "def combine(first, second):\n    return first + second\n", {"first": "left", "second": "right"}),
]


def lex(text):
    """Python lexical units with byte-independent Python character offsets.

    Discard whitespace for matching, retain source offsets for exact rendering.
    Incomplete strings and bracketed expressions are allowed; their completed
    tokens are still usable. No AST of future generated text is constructed.
    """
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result = []
    ignored = {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.INDENT,
               tokenize.DEDENT, tokenize.NL, tokenize.NEWLINE, tokenize.COMMENT}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type in ignored or not tok.string.strip():
                continue
            a = offsets[min(tok.start[0] - 1, len(offsets) - 1)] + tok.start[1]
            b = offsets[min(tok.end[0] - 1, len(offsets) - 1)] + tok.end[1]
            result.append((tok.type, tok.string, a, b))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return result


def declared_parameters(prefix):
    """Conservative prototype: simple def headers without annotations/defaults."""
    matches = list(re.finditer(r"\bdef\s+([A-Za-z_]\w*)\s*\(([^()]*)\)\s*(?:->[^:\n]+)?\s*:", prefix))
    if not matches:
        return set()
    header = matches[-1]
    try:
        fn = ast.parse(header.group(0) + "\n    pass\n").body[0]
    except SyntaxError:
        return set()
    return {arg.arg for arg in fn.args.args}


def signature(tokens, dynamic):
    """Canonical equality pattern; this is classical parameterized matching."""
    names = {}
    out = []
    for kind, value, *_ in tokens:
        if kind == tokenize.NAME and value in dynamic:
            if value not in names:
                names[value] = len(names)
            out.append(("parameter", names[value]))
        else:
            out.append((kind, value))
    return tuple(out)


def unify(source, query, source_vars, query_vars):
    forward, reverse = {}, {}
    for s, q in zip(source, query):
        sd = s[0] == tokenize.NAME and s[1] in source_vars
        qd = q[0] == tokenize.NAME and q[1] in query_vars
        if sd != qd:
            return None
        if sd:
            if s[1] in forward and forward[s[1]] != q[1]:
                return None
            if q[1] in reverse and reverse[q[1]] != s[1]:
                return None
            forward[s[1]], reverse[q[1]] = q[1], s[1]
        elif s[:2] != q[:2]:
            return None
    return forward


class BindingProposer:
    def __init__(self, source, tokenizer, rewrite=True, windows=range(3, 25), cap=8):
        self.source, self.tokenizer, self.rewrite = source, tokenizer, rewrite
        self.tokens = lex(source)
        fn = ast.parse(source).body[0]
        self.params = {a.arg for a in fn.args.args}
        self.windows = tuple(windows)
        self.cap = cap
        self.index = defaultdict(list)
        for end in range(1, len(self.tokens) + 1):
            for n in self.windows:
                if end >= n:
                    sig = signature(self.tokens[end-n:end], self.params)
                    self.index[(n, sig)].append(end)

    def propose(self, generated_ids, generated_text=None):
        self.last_mapping = None
        text = generated_text if generated_text is not None else self.tokenizer.decode(
            generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # Do not extend a partially emitted name, number or string.
        if not text or text[-1].isalnum() or text[-1] in "_'\"":
            return []
        query = lex(text)
        if not query or query[-1][3] != len(text.rstrip()):
            return []
        params = declared_parameters(text)
        for n in reversed(self.windows):
            if len(query) < n:
                continue
            sig = signature(query[-n:], params)
            for end in self.index.get((n, sig), []):
                mapping = unify(self.tokens[end-n:end], query[-n:], self.params, params)
                if mapping is None:
                    continue
                start = self.tokens[end - 1][3]
                cursor = start
                pieces = []
                for tok in self.tokens[end:]:
                    kind, value, a, b = tok
                    if kind == tokenize.NAME and value in self.params:
                        if value not in mapping:
                            break
                        replacement = mapping[value] if self.rewrite else value
                    else:
                        replacement = value
                    pieces.extend((self.source[cursor:a], replacement))
                    cursor = b
                    if len(pieces) >= 128:
                        break
                # Retokenize with the visible generated prefix, then require
                # exact agreement with its committed token IDs. Never replace
                # a token that was already emitted by the target.
                extension = "".join(pieces)
                trailing = text[query[-1][3]:]
                if trailing:
                    if not extension.startswith(trailing):
                        continue
                    extension = extension[len(trailing):]
                ids = self.tokenizer.encode(text + extension, add_special_tokens=False)
                if ids[:len(generated_ids)] != generated_ids:
                    continue
                draft = ids[len(generated_ids):len(generated_ids)+self.cap]
                if draft:
                    self.last_mapping = mapping
                    return draft
        return []


class ExactProposer:
    """Python exact BPE scan baseline, not a reproduction of compiled vLLM KMP."""
    def __init__(self, source, tokenizer, min_n=2, max_n=8, cap=8):
        self.initial = tokenizer.encode(source, add_special_tokens=False)
        self.min_n, self.max_n, self.cap = min_n, max_n, cap

    def propose(self, generated_ids, generated_text=None):
        tokens = self.initial + generated_ids
        # Earliest occurrence tie break; longest match wins. This Python scan
        # is for coverage, not a fair microbenchmark against compiled KMP.
        for n in range(min(self.max_n, len(generated_ids)), self.min_n-1, -1):
            suffix = tokens[-n:]
            for start in range(len(tokens)-n):
                if tokens[start:start+n] == suffix:
                    return tokens[start+n:min(start+n+self.cap, len(tokens))]
        return []


def replay(ids, proposer):
    position, steps, accepted, drafted = 0, 0, 0, 0
    times = []
    while position < len(ids):
        start = time.perf_counter_ns()
        draft = proposer.propose(ids[:position])
        times.append((time.perf_counter_ns() - start) / 1000)
        count = 0
        for a, b in zip(draft, ids[position:]):
            if a != b:
                break
            count += 1
        accepted += count
        drafted += len(draft)
        position += min(count + 1, len(ids) - position)
        steps += 1
    return {"steps": steps, "accepted": accepted, "drafted": drafted,
            "tokens": len(ids), "tokens_per_oracle_step": len(ids)/steps,
            "proposer_us_median": statistics.median(times),
            "proposer_us_total": sum(times)}


def run():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(ROOT / "model", local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        ROOT / "model", local_files_only=True, dtype=torch.float32,
        attn_implementation="sdpa").eval()
    records = []
    for name, source, renames in CASES:
        request = "Rewrite this Python function. "
        if name == "logic_change":
            request += "Change addition to subtraction, and "
        if renames:
            request += "rename " + ", ".join(f"{k} to {v}" for k,v in renames.items())
            request += ". Preserve all other code exactly, including formatting."
        else:
            request += "Return the identical function without changes."
        request += " Return only Python code, with no explanation.\n```python\n" + source + "```"
        inputs = tok.apply_chat_template([{"role":"user", "content":request}],
            add_generation_prompt=True, return_tensors="pt")
        started = time.perf_counter()
        with torch.inference_mode():
            result = model.generate(inputs, attention_mask=torch.ones_like(inputs),
                do_sample=False, max_new_tokens=192, use_cache=True,
                pad_token_id=tok.eos_token_id)
        elapsed = time.perf_counter() - started
        ids = result[0, inputs.shape[1]:].tolist()
        record = {"name":name, "source":source, "requested_renames":renames,
                  "prompt":request, "ids":ids, "seconds":elapsed,
                  "output":tok.decode(ids, skip_special_tokens=True)}
        records.append(record)
        (ROOT / "traces.json").write_text(json.dumps(records,indent=2))
        print(json.dumps({"case":name,"tokens":len(ids),"seconds":elapsed}), flush=True)
    output = {"scope":"Exploratory greedy trace replay, not vLLM latency results",
              "model":json.loads((ROOT/"model/source.json").read_text()),
              "torch":torch.__version__,"transformers":transformers.__version__,
              "hardware":platform.processor(),"threads":4,"records":[]}
    for record in records:
        name, source, ids = record["name"], record["source"], record["ids"]
        row = {"name":name,"output":record["output"]}
        for method,proposer in [("exact",ExactProposer(source,tok)),
                                ("binding",BindingProposer(source,tok)),
                                ("structure_without_rewrite",BindingProposer(source,tok,rewrite=False))]:
            row[method] = replay(ids,proposer)
        output["records"].append(row)
        print(json.dumps(row),flush=True)
    (ROOT / "pilot_results.json").write_text(json.dumps(output,indent=2))


if __name__ == "__main__":
    run()
