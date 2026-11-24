"""Measured GPU inference with CPU drafting, fixed constructed tasks and controls."""
import os
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,"/var/tmp/work-mshanthappa/aria-style-lab/deps")
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'

import argparse
import hashlib
import json
import random
import time
import platform
import torch
import transformers
from transformers import AutoTokenizer,AutoModelForCausalLM
from proposers import IndexedNgram,EditRules
from tasks import make_tasks


def run_request(model,tok,task,method,cap=8,limit=512,trace=False):
    torch.cuda.synchronize()
    start=time.perf_counter()
    text=tok.apply_chat_template([{'role':'user','content':task['prompt']}],
                                tokenize=False,add_generation_prompt=True)+task['prefix']
    full=tok.encode(text,add_special_tokens=False)
    setup_start=time.perf_counter()
    if method=='autoregressive':proposer=None
    elif method=='ngram':proposer=IndexedNgram(full,cap=cap)
    else:proposer=EditRules(task['source'],tok,full,task['prefix'],cap=cap,mode=method)
    setup_seconds=time.perf_counter()-setup_start
    inputs=torch.tensor([full],dtype=torch.long,device='cuda')
    prefill_start=time.perf_counter()
    prediction=model(inputs[:,:-1],use_cache=True,logits_to_keep=1)
    cache=prediction.past_key_values
    torch.cuda.synchronize()
    prefill_seconds=time.perf_counter()-prefill_start
    output=[]
    tail=full[-1]
    steps=accepted=drafted=0
    proposal_seconds=verify_seconds=0.0
    first_token_seconds=None
    step_trace=[]
    while len(output)<limit:
        t=time.perf_counter()
        draft=[] if proposer is None else proposer.propose(output)
        draft=draft[:max(0,limit-len(output)-1)]
        proposal_seconds+=time.perf_counter()-t
        old_len=cache.get_seq_length()
        block=torch.tensor([[tail]+draft],dtype=torch.long,device='cuda')
        t=time.perf_counter()
        prediction=model(block,past_key_values=cache,use_cache=True)
        choices=prediction.logits[0].argmax(dim=-1).tolist()
        cache=prediction.past_key_values
        verify_seconds+=time.perf_counter()-t
        count=0
        for a,b in zip(draft,choices):
            if a!=b:break
            count+=1
        committed=draft[:count]+[choices[count]]
        stop=tok.eos_token_id in committed
        if stop:committed=committed[:committed.index(tok.eos_token_id)+1]
        cache.crop(old_len+1+count)
        if trace:step_trace.append({'output_position':len(output),'draft':draft,
            'accepted':min(count,len(committed)),'committed':committed})
        output.extend(committed)
        tail=committed[-1]
        steps+=1
        accepted+=min(count,len(committed))
        drafted+=len(draft)
        if first_token_seconds is None:first_token_seconds=time.perf_counter()-start
        if stop:break
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-start
    result={'ids':output,'tokens':len(output),'input_tokens':len(full),'steps':steps,
            'accepted':accepted,'drafted':drafted,'seconds':elapsed,
            'tokens_per_second':len(output)/elapsed,'setup_seconds':setup_seconds,
            'prefill_seconds':prefill_seconds,'proposal_seconds':proposal_seconds,
            'verify_seconds':verify_seconds,'first_token_seconds':first_token_seconds,
            'hit_output_limit':len(output)>=limit,'step_trace':step_trace}
    if proposer is not None:result['proposer']=proposer.diagnostics()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['small','7b'],default='small')
    parser.add_argument('--repeats',type=int,default=5)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available(),'CUDA GPU required'
    model_dir=HERE/'model_7b' if args.model=='7b' else HERE.parent/'model'
    tasks=make_tasks()
    if args.smoke:tasks=[tasks[0],tasks[2],tasks[4],tasks[6]]
    methods=['autoregressive','ngram','heuristic','rules']
    dest=HERE/(args.model+('_smoke' if args.smoke else '_measured'))
    dest.mkdir(exist_ok=True)
    protocol={'model':json.loads((model_dir/'source.json').read_text()),
        'dtype':'bfloat16','gpu':torch.cuda.get_device_name(0),
        'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'cpu':'Intel Xeon Silver 4210R','torch_cpu_threads':1,
        'batch_size':1,'concurrency':1,'cap':8,'ngram_min':2,'ngram_max':8,
        'max_new_tokens':512,'repetitions':args.repeats,'seed':42,
        'timing_includes':['initial_tokenization','index_build','prefill','CPU_proposals','GPU_verification','KV_rollback','synchronization'],
        'timing_excludes':['model_load','warmup','network','request_queue'],
        'source_hashes':{f:hashlib.sha256((HERE/f).read_bytes()).hexdigest() for f in ['benchmark.py','proposers.py','tasks.py']},
        'tasks':tasks,'methods':methods,
        'scope':'Constructed prefix-conditioned code edits; custom Transformers loop, not vLLM; batch-one throughput only'}
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
    tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,
        dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
    protocol['parameters']=sum(p.numel() for p in model.parameters())
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2))
    records=[]
    references={}
    with torch.inference_mode():
        # Warm each code path, then establish untimed plain-greedy references.
        for method in methods:run_request(model,tok,tasks[0],method,limit=32)
        for task in tasks:
            ref=run_request(model,tok,task,'autoregressive')
            references[task['name']]=ref['ids']
            print(json.dumps({'reference':task['name'],'tokens':ref['tokens']}),flush=True)
        # Independent generation API checks the custom ordinary decoding loop.
        independent=[]
        for task in tasks[:2]:
            text=tok.apply_chat_template([{'role':'user','content':task['prompt']}],
                tokenize=False,add_generation_prompt=True)+task['prefix']
            inp=tok(text,return_tensors='pt').to('cuda')
            out=model.generate(**inp,do_sample=False,max_new_tokens=512,
                use_cache=True,pad_token_id=tok.eos_token_id)
            ids=out[0,inp.input_ids.shape[1]:].tolist()
            independent.append({'case':task['name'],'matches_generate':ids==references[task['name']]})
        (dest/'independent_checks.json').write_text(json.dumps(independent,indent=2))
        jobs=[(rep,task,method) for rep in range(args.repeats) for task in tasks for method in methods]
        random.Random(42).shuffle(jobs)
        for rep,task,method in jobs:
            row=run_request(model,tok,task,method,trace=rep==0)
            row.update({'repeat':rep,'case':task['name'],'kind':task['kind'],'method':method,
                'matches_reference':row['ids']==references[task['name']]})
            records.append(row)
            # Outputs and logging are outside the timed region.
            (dest/'results.json').write_text(json.dumps({'records':records,
                'references':references},indent=2))
            print(json.dumps({k:row[k] for k in ['repeat','case','method','tokens','steps','seconds','matches_reference']}),flush=True)
    texts={task['name']:task['prefix']+tok.decode(references[task['name']],skip_special_tokens=True,
        clean_up_tokenization_spaces=False) for task in tasks}
    (dest/'reference_outputs.json').write_text(json.dumps(texts,indent=2))
    (dest/'completed.json').write_text(json.dumps({'records':len(records),
        'all_outputs_match':all(x['matches_reference'] for x in records),
        'independent_checks':independent},indent=2))


if __name__=='__main__':main()
