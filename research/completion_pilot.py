"""Prefix-conditioned completion pilot; retain ordinary-chat failures separately."""
from pilot import *


def replay_after_prefix(ids, prefix_length, proposer):
    position, steps, accepted, drafted = prefix_length, 0, 0, 0
    times = []
    while position < len(ids):
        begin = time.perf_counter_ns()
        proposal = proposer.propose(ids[:position])
        times.append((time.perf_counter_ns()-begin)/1000)
        count = 0
        for a,b in zip(proposal,ids[position:]):
            if a != b:
                break
            count += 1
        accepted += count
        drafted += len(proposal)
        position += min(count+1,len(ids)-position)
        steps += 1
    return {"steps":steps,"accepted":accepted,"drafted":drafted,
            "tokens":len(ids)-prefix_length,
            "tokens_per_oracle_step":(len(ids)-prefix_length)/steps,
            "proposer_us_total":sum(times)}


def run_completion():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(ROOT/"model",local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/"model",local_files_only=True,
        dtype=torch.float32,attn_implementation="sdpa").eval()
    records=[]
    for name,source,renames in CASES:
        original_header=source.splitlines()[0]
        header=original_header
        for before,after in renames.items():
            header=re.sub(r"\b"+re.escape(before)+r"\b",after,header)
        prefix="```python\n"+header+"\n    "
        prompt=("Complete the function whose header is supplied below. "
                "Use the supplied argument names throughout the body. "
                "Follow this reference implementation and preserve its logic.\n```python\n"
                +source+"```\nReturn only the completed code.")
        if name=="logic_change":
            prompt+=" Change addition to subtraction."
        chat=tok.apply_chat_template([{"role":"user","content":prompt}],
                                    tokenize=False,add_generation_prompt=True)
        inputs=tok(chat+prefix,return_tensors="pt").input_ids
        start=time.perf_counter()
        with torch.inference_mode():
            result=model.generate(inputs,attention_mask=torch.ones_like(inputs),
                do_sample=False,max_new_tokens=128,use_cache=True,
                pad_token_id=tok.eos_token_id)
        elapsed=time.perf_counter()-start
        continuation=result[0,inputs.shape[1]:].tolist()
        prefix_ids=tok.encode(prefix,add_special_tokens=False)
        ids=prefix_ids+continuation
        row={"name":name,"source":source,"prefix":prefix,"prompt":prompt,
             "ids":ids,"prefix_ids":prefix_ids,"new_tokens":len(continuation),
             "seconds":elapsed,"output":tok.decode(ids,skip_special_tokens=True)}
        for method,proposer in [("exact",ExactProposer(source,tok)),
                ("binding",BindingProposer(source,tok)),
                ("structure_without_rewrite",BindingProposer(source,tok,rewrite=False))]:
            row[method]=replay_after_prefix(ids,len(prefix_ids),proposer)
        records.append(row)
        (ROOT/"completion_results.json").write_text(json.dumps(records,indent=2))
        print(json.dumps({k:v for k,v in row.items() if k not in {"ids","source","prompt","prefix_ids"}}),flush=True)


if __name__=="__main__":
    run_completion()
