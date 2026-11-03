import unittest
from masked_ttt.output_guard import decide,same_text


class OutputGuardTests(unittest.TestCase):
    def choose(self,first,second):
        return decide('300','320',first,second,minimum_confidence=.65,minimum_label_mass=.2)

    def test_adapted_requires_consensus_in_both_orders(self):
        favors_b=dict(A=.1,B=.8,U=.1,label_mass=.7)
        favors_a=dict(A=.8,B=.1,U=.1,label_mass=.7)
        self.assertTrue(self.choose(favors_b,favors_a)['use_adapted'])
        self.assertFalse(self.choose(favors_b,favors_b)['use_adapted'])
        self.assertFalse(self.choose(dict(favors_b,label_mass=.1),favors_a)['use_adapted'])
        self.assertFalse(self.choose(dict(A=.2,B=.6,U=.2,label_mass=.7),favors_a)['use_adapted'])

    def test_invalid_or_missing_scores_keep_baseline(self):
        good=dict(A=.8,B=.1,U=.1,label_mass=.7)
        for bad in (None,dict(A=float('nan'),B=.1,U=.1,label_mass=.7),
                    dict(A=.8,B=.8,U=.1,label_mass=.7)):
            self.assertFalse(self.choose(bad,good)['use_adapted'])

    def test_identical_answer_needs_no_judge(self):
        self.assertTrue(same_text(' 320  万元 ','320 万元'))
        self.assertTrue(decide('320 万元','320  万元',None,None,
                               minimum_confidence=.65,minimum_label_mass=.2)['use_adapted'])


if __name__=='__main__':unittest.main()
