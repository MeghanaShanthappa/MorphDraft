"""CPU proposers. Neither instruction text nor future target tokens reach rule learning."""
import ast
import copy
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from edit_rule_proof import Rule


class IndexedNgram:
    def __init__(self, initial, min_n=2, max_n=8, cap=8):
        self.min_n,self.max_n,self.cap=min_n,max_n,cap
        self.tokens=[]
        self.index={}
        self.observed=0
        self.extend(initial)

    def extend(self, ids):
        for value in ids:
            self.tokens.append(value)
            end=len(self.tokens)
            for n in range(self.min_n,min(self.max_n,end)+1):
                self.index.setdefault(tuple(self.tokens[end-n:end]),end-n)

    def lookup(self, context):
        for n in range(min(self.max_n,len(context)),self.min_n-1,-1):
            at=self.index.get(tuple(context[-n:]))
            if at is not None and at+n<len(self.tokens):
                return self.tokens[at+n:at+n+self.cap]
        return []

    def propose(self, output):
        self.extend(output[self.observed:])
        self.observed=len(output)
        return self.lookup(self.tokens)

    def diagnostics(self):
        return {"index_entries":len(self.index)}


def assignments(tree):
    # Initial prototype: straight-line assignments in a top-level function.
    funcs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    if not funcs:
        return []
    return [n for n in funcs[0].body if isinstance(n,ast.Assign)
            and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)
            and isinstance(n.value,ast.Call)]


def family(node):
    return ast.dump(node.value.func,include_attributes=False)


def dump(node):
    return ast.dump(node,include_attributes=False)


class EditRules:
    """Generalize two completed source/committed-output pairs per call family.

    Alignment uses unique assignment names, not reference target completions.
    This restricted workload tests the inference mechanism; general edit
    alignment and concrete-syntax preservation are outside this prototype.
    """
    def __init__(self, source, tokenizer, initial, generated_prefix,
                 min_n=2,max_n=8,cap=8, mode="rules"):
        self.tok,self.prefix,self.mode=tokenizer,generated_prefix,mode
        self.min_n,self.max_n,self.cap=min_n,max_n,cap
        self.source=source
        self.tree=ast.parse(source)
        self.source_nodes={n.targets[0].id:n for n in assignments(self.tree)}
        self.fallback=IndexedNgram(initial,min_n,max_n,cap)
        self.seen=set()
        self.pairs=defaultdict(list)
        self.rules={}
        self.rewritten_index=None
        self.last_rewritten=None
        self.last_boundary=-1
        self.updates=0
        self.rule_proposals=0
        self.proposal_calls=0
        self.first_rule_after_tokens=None

    def derive(self, pairs):
        if self.mode=="heuristic":
            # Strong cheap control: one observed callee change plus appended
            # literal keywords. Preserve argument list and assignment target.
            s,t=pairs[-1]
            a,b=s.value,t.value
            if len(a.args)!=len(b.args) or any(dump(x)!=dump(y) for x,y in zip(a.args,b.args)):
                return None
            if len(b.keywords)<len(a.keywords):return None
            for x,y in zip(a.keywords,b.keywords):
                if x.arg!=y.arg or dump(x.value)!=dump(y.value):return None
            extra=b.keywords[len(a.keywords):]
            if any(not isinstance(k.value,ast.Constant) for k in extra):return None
            return (copy.deepcopy(b.func),copy.deepcopy(extra))
        if len(pairs)<2:return None
        s1,t1=pairs[-2]
        s2,t2=pairs[-1]
        try:
            rule=Rule(s1,s2,t1,t2)
        except ValueError:
            return None
        # A generalization must agree with every observed matching example.
        for s,t in pairs:
            predicted=rule.apply(s)
            if predicted is not None and dump(predicted)!=dump(t):return None
        return rule

    def apply(self, rule, source):
        if self.mode=="heuristic":
            out=copy.deepcopy(source)
            out.value.func=copy.deepcopy(rule[0])
            out.value.keywords.extend(copy.deepcopy(rule[1]))
            return out
        return rule.apply(source)

    def observe_completed(self, output):
        text=self.prefix+self.tok.decode(output,skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        # Only newline-terminated text is available to the parser. Partial
        # identifiers, arguments and statements cannot provide observations.
        boundary=text.rfind("\n")
        if boundary<=self.last_boundary:return
        self.last_boundary=boundary
        complete=text[:boundary+1]
        if complete.startswith("```python\n"):complete=complete[len("```python\n"):]
        elif complete.startswith("```\n"):complete=complete[4:]
        complete=complete.split("```")[0]
        try:tree=ast.parse(complete)
        except (SyntaxError,ValueError):return
        changed=False
        for target in assignments(tree):
            name=target.targets[0].id
            if name in self.seen or name not in self.source_nodes:continue
            self.seen.add(name)
            source=self.source_nodes[name]
            f=family(source)
            self.pairs[f].append((source,target))
            if all(dump(s)==dump(t) for s,t in self.pairs[f]):continue
            rule=self.derive(self.pairs[f])
            if rule is None:self.rules.pop(f,None)
            else:self.rules[f]=rule
            changed=True
        if not changed:return
        if not self.rules:
            self.rewritten_index=None
            self.last_rewritten=None
            return
        proposed=copy.deepcopy(self.tree)
        function=next(n for n in proposed.body if isinstance(n,ast.FunctionDef))
        for i,node in enumerate(function.body):
            if isinstance(node,ast.Assign) and isinstance(node.value,ast.Call):
                rule=self.rules.get(family(node))
                if rule is not None:
                    transformed=self.apply(rule,node)
                    if transformed is not None:function.body[i]=transformed
        rewritten=ast.unparse(ast.fix_missing_locations(proposed))+"\n"
        if rewritten!=self.last_rewritten:
            self.last_rewritten=rewritten
            self.rewritten_index=IndexedNgram(self.tok.encode(rewritten,add_special_tokens=False),
                self.min_n,self.max_n,self.cap)
            self.updates+=1
            if self.rules and self.first_rule_after_tokens is None:
                self.first_rule_after_tokens=len(output)

    def propose(self, output):
        self.proposal_calls+=1
        ordinary=self.fallback.propose(output)
        self.observe_completed(output)
        if self.rewritten_index is not None:
            candidate=self.rewritten_index.lookup(self.fallback.tokens)
            if candidate:
                self.rule_proposals+=1
                return candidate
        return ordinary

    def diagnostics(self):
        return {"observed_source_target_pairs":len(self.seen),"rules":len(self.rules),
                "rewritten_index_updates":self.updates,
                "rule_proposals":self.rule_proposals,"proposal_calls":self.proposal_calls,
                "first_rule_after_tokens":self.first_rule_after_tokens,
                "index_entries":len(self.fallback.index)+(len(self.rewritten_index.index) if self.rewritten_index else 0)}
