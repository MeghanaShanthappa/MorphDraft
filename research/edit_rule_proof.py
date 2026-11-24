"""Mechanism proof for the proposed next stage; not an LLM speed experiment.

Joint anti-unification shares holes across source and target templates. Only
completed, already-accepted target calls may be supplied as training examples.
"""
import ast
import copy
import json
from pathlib import Path


class Hole:
    def __init__(self, index):
        self.index=index


def key(node):
    return ast.dump(node,include_attributes=False) if isinstance(node,ast.AST) else repr(node)


class Rule:
    def __init__(self, source_a, source_b, target_a, target_b):
        self.holes={}
        self.source=self.generalize(source_a,source_b)
        self.target=self.generalize(target_a,target_b)
        # A target-only hole cannot be predicted from a new source call.
        def ids(x):
            if isinstance(x,Hole): return {x.index}
            if isinstance(x,tuple): return set().union(*(ids(y) for y in x))
            if isinstance(x,list): return set().union(*(ids(y) for y in x))
            return set()
        if not ids(self.target).issubset(ids(self.source)):
            raise ValueError("Target transformation is underdetermined")

    def generalize(self,a,b):
        if isinstance(a,ast.AST) and type(a) is type(b):
            return ("node",type(a),[(name,self.generalize(getattr(a,name),getattr(b,name))) for name in a._fields])
        if isinstance(a,list) and isinstance(b,list) and len(a)==len(b):
            return [self.generalize(x,y) for x,y in zip(a,b)]
        if key(a)==key(b):
            return ("literal",copy.deepcopy(a))
        pair=(key(a),key(b))
        if pair not in self.holes:
            self.holes[pair]=len(self.holes)
        return Hole(self.holes[pair])

    def match(self,template,value,bindings):
        if isinstance(template,Hole):
            if template.index in bindings:
                return key(bindings[template.index])==key(value)
            bindings[template.index]=copy.deepcopy(value)
            return True
        if isinstance(template,list):
            return isinstance(value,list) and len(template)==len(value) and all(self.match(a,b,bindings) for a,b in zip(template,value))
        if template[0]=="literal":
            return key(template[1])==key(value)
        _,cls,fields=template
        return type(value) is cls and all(self.match(t,getattr(value,n),bindings) for n,t in fields)

    def build(self,template,bindings):
        if isinstance(template,Hole): return copy.deepcopy(bindings[template.index])
        if isinstance(template,list): return [self.build(t,bindings) for t in template]
        if template[0]=="literal": return copy.deepcopy(template[1])
        _,cls,fields=template
        return cls(**{n:self.build(t,bindings) for n,t in fields})

    def apply(self,source):
        bindings={}
        if not self.match(self.source,source,bindings):
            return None
        return ast.fix_missing_locations(self.build(self.target,bindings))


def expr(text):
    return ast.parse(text,mode="eval").body


def main():
    rule=Rule(expr('legacy_fetch("first", retries=1)'),expr('legacy_fetch("second", retries=1)'),
              expr('client.fetch("first", retries=1, timeout=10)'),
              expr('client.fetch("second", retries=1, timeout=10)'))
    proposal=rule.apply(expr('legacy_fetch("third", retries=1)'))
    assert ast.unparse(proposal)=="client.fetch('third', retries=1, timeout=10)"
    assert rule.apply(expr('unrelated_fetch("third", retries=1)')) is None
    assert rule.apply(expr('legacy_fetch("third", retries=9)')) is None
    alias=Rule(expr('old("a", "a")'),expr('old("b", "b")'),
               expr('new("a", "a")'),expr('new("b", "b")'))
    assert alias.apply(expr('old("c", "d")')) is None
    assert alias.apply(expr('old("c", "c")')) is not None
    underdetermined=False
    try:
        Rule(expr('old("a")'),expr('old("b")'),expr('new("x")'),expr('new("y")'))
    except ValueError:
        underdetermined=True
    assert underdetermined
    result={"scope":"Constructed AST mechanism proof only; no target model",
            "observed_pairs":[['legacy_fetch("first", retries=1)','client.fetch("first", retries=1, timeout=10)'],
                              ['legacy_fetch("second", retries=1)','client.fetch("second", retries=1, timeout=10)']],
            "unseen_source":'legacy_fetch("third", retries=1)',
            "proposed_continuation":ast.unparse(proposal),"assertions_passed":6}
    Path(__file__).with_name("edit_rule_proof.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
