"""Download a pinned checkpoint; no credentials are stored in this repository."""
import argparse
import json
from pathlib import Path

MODELS={
    '7b':('Qwen/Qwen2.5-Coder-7B-Instruct','c03e6d358207e414f1eca0bb1891e29f1db0e242'),
    'small':('Qwen/Qwen2.5-Coder-0.5B-Instruct','ea3f2471cf1b1f0db85067f1ef93848e38e88c25'),
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=MODELS,default='7b')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from huggingface_hub import snapshot_download
    repository,revision=MODELS[a.model]
    snapshot_download(repo_id=repository,revision=revision,local_dir=a.output,
                      allow_patterns=['*.json','*.safetensors','*.txt','*.model','LICENSE*','README.md'])
    (a.output/'source.json').write_text(json.dumps({'repository':repository,'revision':revision},indent=2)+'\n')
    print(f'Pinned checkpoint ready at {a.output}')

if __name__=='__main__':main()
