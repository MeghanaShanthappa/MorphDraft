"""Constructed renaming stress test from HumanEval source functions.

This exposes reference solutions in the prompt. It is NOT HumanEval pass@1.
"""
from completion_pilot import *
from verify_benchmark import HybridProposer
import gzip


def select():
    examples=[]
    for line in gzip.decompress((ROOT/"HumanEval.jsonl.gz").read_bytes()).decode().splitlines():
        raw=json.loads(line)
        tree=ast.parse(raw["prompt"]+raw["canonical_solution"])
        fns=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
        if not fns:
            continue
        fn=fns[0]
        if fn.body and isinstance(fn.body[0],ast.Expr) and isinstance(fn.body[0].value,ast.Constant) and isinstance(fn.body[0].value.value,str):
            fn.body=fn.body[1:]
        if not fn.args.args or any(isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Lambda)) for body in fn.body for n in ast.walk(body)):
            continue
        source=ast.unparse(fn)+"\n"
        if len(lex(source))>120 or len(source.splitlines())<2:
            continue
        names={a.arg:f"parameter_{i}" for i,a in enumerate(fn.args.args)}
        header=source.splitlines()[0]
        for old,new in names.items():
            header=re.sub(r"\b"+re.escape(old)+r"\b",new,header)
        # Pilot parser supports only observed simple function headers.
        if set(names.values())!=declared_parameters(header):
            continue
        examples.append({"task_id":raw["task_id"],"source":source,
                         "prefix":"```python\n"+header+"\n    ","renames":names})
        if len(examples)==16:
            break
    return examples


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    tok=AutoTokenizer.from_pretrained(ROOT/"model",local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/"model",local_files_only=True,
        dtype=torch.float32,attn_implementation="sdpa").eval()
    cases=select()
    (ROOT/"corpus_selection.json").write_text(json.dumps(cases,indent=2))
    records=[]
    with torch.inference_mode():
        for case in cases:
            prompt=("Complete the function whose header is supplied below. Use the supplied "
                    "argument names throughout the body. Follow this reference implementation "
                    "and preserve its logic.\n```python\n"+case["source"]+
                    "```\nReturn only the completed code.")
            chat=tok.apply_chat_template([{"role":"user","content":prompt}],
                tokenize=False,add_generation_prompt=True)
            inputs=tok(chat+case["prefix"],return_tensors="pt").input_ids
            result=model.generate(inputs,attention_mask=torch.ones_like(inputs),
                do_sample=False,max_new_tokens=128,use_cache=True,
                pad_token_id=tok.eos_token_id)
            continuation=result[0,inputs.shape[1]:].tolist()
            prefix_ids=tok.encode(case["prefix"],add_special_tokens=False)
            ids=prefix_ids+continuation
            row={**case,"ids":ids,"prefix_ids":prefix_ids,"prompt":prompt,
                 "output":tok.decode(ids,skip_special_tokens=True)}
            for name,proposer in [("exact",ExactProposer(case["source"],tok)),
                    ("binding",BindingProposer(case["source"],tok)),
                    ("hybrid",HybridProposer(case["source"],tok))]:
                row[name]=replay_after_prefix(ids,len(prefix_ids),proposer)
            records.append(row)
            (ROOT/"corpus_results.json").write_text(json.dumps(records,indent=2))
            print(json.dumps({"task":case["task_id"],"tokens":len(continuation),
                "steps":{k:row[k]["steps"] for k in ["exact","binding","hybrid"]}}),flush=True)


if __name__=="__main__":
    main()
