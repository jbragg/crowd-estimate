import unittest
from prob import *

class ProbTest(unittest.TestCase):
    def setUp(self):
        self.dai_module = ProbabilityDai()
        self.mdp_module = ProbabilityMDP()

    def test_allprobs(self):
        #self.dai_module.allprobs(s,d)
        pass

    def test_allprobs_ddifficulty(self):
        pass

    def test_allprobs_dskill(self):
        pass

    def test_dai_transform_skill(self):
        s = np.array([0.6])
        res = self.dai_module.transform_skill(s)
        self.assertEqual(1/s, res)

    def test_dai_prob_correct(self):
        s = np.array([0.6])
        d = np.array([0.9])
        expected = np.array([0.5 * (1+np.power(1-d,s))])
        res = self.dai_module.prob_correct(1/s,d)
        self.assertEqual(expected, res)

    def test_dai_prob_correct_ddifficulty(self):
        s = np.array([0.6])
        d = np.array([0.9])
        expected = np.array([-0.5 * s * np.power(1-d,s-1)])
        res = self.dai_module.prob_correct_ddifficulty(1/s,d)
        self.assertEqual(expected, res)

    def test_dai_prob_correct_dskill(self):
        s = np.array([0.6])
        d = np.array([0.9])
        expected = np.array([0.5 * np.power(1-d,s) * np.log(1-d)])
        res = self.dai_module.prob_correct_dskill(1/s,d)
        self.assertEqual(expected, res)
