import unittest
import ast
from proposers import IndexedNgram,EditRules
from tasks import make_tasks


class ByteTokenizer:
    def encode(self,text,**kwargs):return list(text.encode())
    def decode(self,ids,**kwargs):return bytes(ids).decode()


class ProposerTests(unittest.TestCase):
    def test_index_matches_exhaustive_reference(self):
        import random
        rng=random.Random(13)
        for _ in range(300):
            initial=[rng.randrange(8) for i in range(40)]
            output=[rng.randrange(8) for i in range(20)]
            p=IndexedNgram(initial,min_n=2,max_n=6,cap=8)
            for size in [0,1,4,9,20]:
                actual=p.propose(output[:size])
                all_tokens=initial+output[:size]
                expected=[]
                for n in range(6,1,-1):
                    for j in range(len(all_tokens)-n):
                        if all_tokens[j:j+n]==all_tokens[-n:]:
                            expected=all_tokens[j+n:j+n+8]
                            break
                    if expected:break
                self.assertEqual(actual,expected)

    def test_rule_requires_two_completed_observations(self):
        task=make_tasks()[0]
        tok=ByteTokenizer()
        p=EditRules(task['source'],tok,tok.encode(task['prompt']+task['prefix']),task['prefix'])
        one="    result_0 = client.fetch('record_0', retries=1, timeout=10)\n"
        p.propose(tok.encode(one))
        self.assertEqual(len(p.rules),0)
        partial=one+"    result_1 = client.fetch('record_1', retries=1, timeout=10)"
        p.propose(tok.encode(partial))
        self.assertEqual(len(p.rules),0)
        p.propose(tok.encode(partial+'\n'))
        self.assertEqual(len(p.rules),1)
        self.assertIn("result_7 = client.fetch('record_7', retries=1, timeout=10)",p.last_rewritten)

    def test_rule_reuses_reordered_arguments(self):
        task=make_tasks()[2]
        tok=ByteTokenizer()
        p=EditRules(task['source'],tok,tok.encode(task['prompt']+task['prefix']),task['prefix'])
        observed="".join(f"    result_{i} = client.send('message_{i}', 'account_{i}', retries=1, secure=True)\n" for i in range(2))
        p.propose(tok.encode(observed))
        self.assertEqual(len(p.rules),1)
        self.assertIn("result_7 = client.send('message_7', 'account_7', retries=1, secure=True)",p.last_rewritten)

    def test_unchanged_code_does_not_create_rules(self):
        task=make_tasks()[4]
        tok=ByteTokenizer()
        p=EditRules(task['source'],tok,tok.encode(task['prompt']+task['prefix']),task['prefix'])
        p.propose(tok.encode('\n'.join(task['source'].splitlines()[1:3])+'\n'))
        self.assertEqual(len(p.rules),0)
        self.assertIsNone(p.rewritten_index)

    def test_conflicting_observed_edit_retires_rule(self):
        task=make_tasks()[0]
        tok=ByteTokenizer()
        p=EditRules(task['source'],tok,tok.encode(task['prompt']+task['prefix']),task['prefix'])
        observed="".join(f"    result_{i} = client.fetch('record_{i}', retries=1, timeout=10)\n" for i in range(2))
        p.propose(tok.encode(observed))
        self.assertEqual(len(p.rules),1)
        p.propose(tok.encode(observed+"    result_2 = legacy_fetch('record_2', retries=1)\n"))
        self.assertEqual(len(p.rules),0)
        self.assertIsNone(p.rewritten_index)

    def test_heuristic_control_handles_keyword_deletion_after_one_example(self):
        task=make_tasks()[0]
        tok=ByteTokenizer()
        p=EditRules(task['source'],tok,tok.encode(task['prompt']+task['prefix']),
                    task['prefix'],mode='heuristic')
        p.propose(tok.encode("    result_0 = client.fetch('record_0', timeout=10)\n"))
        self.assertEqual(len(p.rules),1)
        self.assertIn("result_7 = client.fetch('record_7', timeout=10)",p.last_rewritten)
        self.assertNotIn('retries',p.last_rewritten)


if __name__=='__main__':unittest.main()
