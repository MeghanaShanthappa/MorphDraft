"""Meaningful matcher and tokenizer invariants, independent of model quality."""
from pilot import *
import itertools
import unittest


class MatcherTests(unittest.TestCase):
    def test_alias_identity_is_preserved(self):
        self.assertNotEqual(signature(lex("x+x"),{"x","y"}),
                            signature(lex("a+b"),{"a","b"}))
        self.assertEqual(signature(lex("x+x"),{"x","y"}),
                         signature(lex("b+b"),{"a","b"}))

    def test_bijection_matches_independent_bruteforce(self):
        rng=random.Random(924)
        for _ in range(1000):
            source=[rng.choice(["x","y","+","-"]) for i in range(8)]
            query=[rng.choice(["a","b","+","-"]) for i in range(8)]
            expected=any([dict(zip(["x","y"],p)).get(s,s) for s in source]==query
                         for p in itertools.permutations(["a","b"]))
            # Space-separated lexemes avoid joining repeated operators.
            actual=unify(lex(" ".join(source)),lex(" ".join(query)),
                         {"x","y"},{"a","b"}) is not None
            self.assertEqual(expected,actual)

    def test_keywords_and_operators_are_literal(self):
        self.assertIsNone(unify(lex("return x+x"),lex("return a-a"),{"x"},{"a"}))

    def test_literals_are_not_renamed(self):
        self.assertIsNone(unify(lex("x+'old'"),lex("a+'new'"),{"x"},{"a"}))

    def test_observed_annotations(self):
        self.assertEqual(declared_parameters("def f(values: list[int], count: int) -> float:"),
                         {"values","count"})
        self.assertEqual(declared_parameters("def f(unfinished"),set())

    def test_substitution_changes_real_bpe_length(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/"model",local_files_only=True)
        source="def f(x):\n    return x + x + x + x\n"
        prefix="def f(very_long_new_identifier):\n    return very_long_new_identifier +"
        ids=tok.encode(prefix,add_special_tokens=False)
        p=BindingProposer(source,tok)
        draft=p.propose(ids)
        self.assertTrue(draft)
        text=tok.decode(ids+draft,clean_up_tokenization_spaces=False)
        self.assertTrue(text.startswith(prefix))
        self.assertIn("very_long_new_identifier",text[len(prefix):])
        self.assertNotEqual(len(tok.encode("x",add_special_tokens=False)),
                            len(tok.encode("very_long_new_identifier",add_special_tokens=False)))

    def test_partial_identifier_abstains(self):
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(ROOT/"model",local_files_only=True)
        p=BindingProposer("def f(x):\n    return x+x\n",tok)
        self.assertEqual(p.propose(tok.encode("def f(new_name):\n    return new_na",add_special_tokens=False)),[])


if __name__=="__main__":
    unittest.main()
