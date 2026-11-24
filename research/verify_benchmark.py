"""Real greedy CPU draft/verify pilot, with KV rollback and output equality checks.

This single-process Transformers benchmark is not vLLM performance evidence.
"""
from completion_pilot import *


class HybridProposer:
    def __init__(self, source, tokenizer):
        self.binding=BindingProposer(source,tokenizer)
        self.exact=ExactProposer(source,tokenizer)

    def propose(self, ids):
        candidate=self.binding.propose(ids)
        mapping=self.binding.last_mapping or {}
        if candidate and any(k!=v for k,v in mapping.items()):
            return candidate
        return self.exact.propose(ids)


def decode(model, tok, full_input_ids, generated_prefix, proposer=None, limit=128):
    import torch
    inputs=torch.tensor([full_input_ids],dtype=torch.long)
    output=[]
    start=time.perf_counter()
    prefill=model(inputs[:,:-1],use_cache=True)
    cache=prefill.past_key_values
    prefill_seconds=time.perf_counter()-start
    tail=full_input_ids[-1]
    steps=accepted=drafted=0
    proposal_seconds=verify_seconds=0.0
    while len(output)<limit:
        t=time.perf_counter()
        draft=[] if proposer is None else proposer.propose(generated_prefix+output)
        draft=draft[:max(0,limit-len(output)-1)]
        proposal_seconds+=time.perf_counter()-t
        old_len=cache.get_seq_length()
        block=torch.tensor([[tail]+draft],dtype=torch.long)
        t=time.perf_counter()
        prediction=model(block,past_key_values=cache,use_cache=True)
        choices=prediction.logits[0].argmax(dim=-1).tolist()
        cache=prediction.past_key_values
        verify_seconds+=time.perf_counter()-t
        count=0
        for a,b in zip(draft,choices):
            if a!=b:
                break
            count+=1
        committed=draft[:count]+[choices[count]]
        cache.crop(old_len+1+count)
        output.extend(committed)
        tail=committed[-1]
        steps+=1
        accepted+=count
        drafted+=len(draft)
        if tok.eos_token_id in committed:
            output=output[:output.index(tok.eos_token_id)+1]
            break
    return {"ids":output,"steps":steps,"accepted":accepted,"drafted":drafted,
            "seconds":time.perf_counter()-start,"prefill_seconds":prefill_seconds,
            "proposal_seconds":proposal_seconds,"verify_seconds":verify_seconds}


def main():
    import torch
    import transformers
    from transformers import AutoModelForCausalLM,AutoTokenizer
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    tok=AutoTokenizer.from_pretrained(ROOT/"model",local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ROOT/"model",local_files_only=True,
        dtype=torch.float32,attn_implementation="sdpa").eval()
    traces=json.loads((ROOT/"completion_results.json").read_text())
    jobs=[(repeat,case,method) for repeat in range(3)
          for case in range(len(traces)) for method in ["autoregressive","exact","hybrid"]]
    random.Random(42).shuffle(jobs)
    records=[]
    expected={}
    with torch.inference_mode():
        # Warm model kernels with a short request; do not include loading/warmup.
        warm=tok("def f(x): return x",return_tensors="pt").input_ids.tolist()[0]
        decode(model,tok,warm,[],limit=4)
        for repeat,i,method in jobs:
            case=traces[i]
            chat=tok.apply_chat_template([{"role":"user","content":case["prompt"]}],
                tokenize=False,add_generation_prompt=True)
            full=tok.encode(chat+case["prefix"],add_special_tokens=False)
            if method=="autoregressive":
                proposer=None
            elif method=="exact":
                proposer=ExactProposer(case["source"],tok)
            else:
                proposer=HybridProposer(case["source"],tok)
            begin=time.perf_counter()
            # Index construction timing is measured separately below. Existing
            # indexes here have already been built for this request.
            result=decode(model,tok,full,case["prefix_ids"],proposer)
            row={"repeat":repeat,"case":case["name"],"method":method,**result}
            records.append(row)
            if case["name"] not in expected:
                expected[case["name"]]=result["ids"]
            row["equal_other_methods"]=expected[case["name"]]==result["ids"]
            (ROOT/"verification_benchmark.json").write_text(json.dumps({
                "scope":"Exploratory eight-case CPU Transformers benchmark",
                "cpu":"Intel Xeon Silver 4210R","threads":4,
                "torch":torch.__version__,"transformers":transformers.__version__,
                "model":json.loads((ROOT/"model/source.json").read_text()),
                "index_construction_excluded":True,"records":records},indent=2))
            print(json.dumps({k:v for k,v in row.items() if k!="ids"}),flush=True)
    assert all(r["equal_other_methods"] for r in records), "Greedy outputs diverged"


if __name__=="__main__":
    main()
