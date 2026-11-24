"""Cheap competing explanation: rename parameters by their header position."""
from completion_pilot import *


class PositionalRenameProposer:
    def __init__(self, source, tok):
        self.source,self.tok=source,tok
        self.original=[a.arg for a in ast.parse(source).body[0].args.args]
        self.fallback=ExactProposer(source,tok)
        self.last_names=None
        self.renamed=None

    def propose(self, ids):
        text=self.tok.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
        headers=list(re.finditer(r"\bdef\s+\w+\s*\([^()]*\)\s*(?:->[^:\n]+)?\s*:",text))
        if not headers:
            return self.fallback.propose(ids)
        try:
            fn=ast.parse(headers[-1].group(0)+"\n    pass\n").body[0]
        except SyntaxError:
            return self.fallback.propose(ids)
        names=tuple(a.arg for a in fn.args.args)
        if len(names)!=len(self.original):
            return self.fallback.propose(ids)
        if names!=self.last_names:
            mapping=dict(zip(self.original,names))
            pieces=[]
            cursor=0
            for kind,value,a,b in lex(self.source):
                pieces.extend((self.source[cursor:a],mapping.get(value,value) if kind==tokenize.NAME else value))
                cursor=b
            pieces.append(self.source[cursor:])
            self.renamed=ExactProposer("".join(pieces),self.tok)
            self.last_names=names
        return self.renamed.propose(ids)


if __name__=="__main__":
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/"model",local_files_only=True)
    output={}
    for filename in ["completion_results.json","corpus_results.json"]:
        records=json.loads((ROOT/filename).read_text())
        rows=[]
        for r in records:
            result=replay_after_prefix(r["ids"],len(r["prefix_ids"]),PositionalRenameProposer(r["source"],tok))
            rows.append({"name":r.get("name",r.get("task_id")),**result})
        output[filename]=rows
        print(filename,'positional_steps',sum(x['steps'] for x in rows),
              'exact_steps',sum(x['exact']['steps'] for x in records))
    (ROOT/"heuristic_control.json").write_text(json.dumps(output,indent=2))
