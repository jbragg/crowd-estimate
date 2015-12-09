import unittest
from inference import *

class InferenceTest(unittest.TestCase):
#   def setUp(self):
#       self.votes = {(345, 2): {'vote': 0},
#                     (1, 2): {'vote': 1},
#                     (1, 3): {'vote': 1}}

    def test_dbeta(self):
        # mode at x = 0.5 (derivative = 0)
        res = dbeta(0.5,2,2)
        expected = 0.0
        self.assertAlmostEqual(expected, res)

        # should be decreasing for x > 0.5
        res = dbeta(0.6,2,2)
        expected = -1.2
        self.assertAlmostEqual(expected, res)

        # should be increasing for x < 0.5
        res = dbeta(0.3,2,2)
        expected = 2.4
        self.assertAlmostEqual(expected, res)

#   #TODO test components of inference module
#   def test_infer(self):
#       pass

#   def test_infer_difficulty_buckets(self):
#       pass
#   
#   #TODO check result vs. expected values

    #TODO bugfix
    def test_dai_all_unknown(self):
        # NOTE vote (worker, question): {'vote': 0/1}
        votes = {(1, 1): {'vote': 0},
                (1,2):{'vote':1},
                      (2, 1): {'vote': 1},
#                     (2, 2): {'vote': 0},
                      (3, 1): {'vote': 1},
                      (3, 2): {'vote': 1},
                      (4, 1): {'vote': 1}}
                      
        workers = {1: {'skill': None},
                        2: {'skill': None},
                        3: {'skill': None},
                        4: {'skill': None}}
        
        questions= {1: {'difficulty': None},
                         2: {'difficulty': None}}
        d_res = self.dai_helper(votes, workers, questions)
        print("Test Dai w/ Difficulties Unknown, Skills Unknown:")
        print(d_res)

    def test_dai_all_known(self):
        votes = {(1, 1): {'vote': 0},
                      (2, 1): {'vote': 1},
                      (2, 2): {'vote': 1}}
                      
        workers = {1: {'skill': 0.6},
                        2: {'skill': 0.7}}
        
        questions= {1: {'difficulty': 0.9},
                         2: {'difficulty': 0.1}}
        d_res = self.dai_helper(votes, workers, questions)

        # NOTE Seth hand-calculated
        expected_q1_post = 0.5078721
        expected_q2_post = 0.9301324
        self.assertAlmostEqual(d_res['posteriors'][1], expected_q1_post)
        self.assertAlmostEqual(d_res['posteriors'][2], expected_q2_post)
        print("Test Dai w/ Difficulties Known, Skills Known:")
        print(d_res)

    def test_dai_skills_known(self):
        votes = {(1, 1): {'vote': 0},
                      (2, 1): {'vote': 1},
                      (2, 2): {'vote': 1}}
                      
        workers = {1: {'skill': 0.6},
                        2: {'skill': 0.5}}
        
        questions= {1: {'difficulty': None},
                         2: {'difficulty': None}}
        d_res = self.dai_helper(votes, workers, questions)

        # Know: P(q1 answer = 1) < 0.5 because worker 1 voted 0
        # and is more skilled than worker 2 who voted 1
        self.assertLess(d_res['posteriors'][1], 0.5)

        # Know: P(q2 answer = 1) = 0.625 because worker 2 voted 1
        # and has skill = 0.5, difficulty prior = 0.5
        # using prob_correct = 1/2 * (1+(1-d)^(1/s)) and
        # prob_incorrect = 1/2 * (1-prob_correct)
        self.assertAlmostEqual(d_res['posteriors'][2], 0.625)

        # Know q1 difficulty should be > q2 difficulty
        # bc workers disagree
        self.assertGreater(d_res['questions'][1], d_res['questions'][2])

        # Know q2 difficulty should remain = to the prior (0.5)
        # because it is independent from other parameters (unless true answer is known)
        expected_q2_difficulty = 0.5
        self.assertAlmostEqual(d_res['questions'][2], expected_q2_difficulty)

        print("Test Dai w/ Difficulties Unknown, Skills Known:")
        print(d_res)

    def test_dai_difficulties_known(self):
        # test dai
        votes = {(1, 1): {'vote': 0},
                      (1, 2): {'vote':1},
                      (2, 1): {'vote': 1},
                      (2, 2): {'vote': 1}}
                      
        workers = {1: {'skill': None},
                        2: {'skill': None}}
        
        questions= {1: {'difficulty': 0.9},
                         2: {'difficulty': 0.1}}
        d_res = self.dai_helper(votes, workers, questions)

        # hand calculated
        expected_q1_post = 0.5
        # TODO BUG program returns 0.99801486... for q2 post
        # My calculation error or bug?
        expected_q2_post = 0.9891009
        expected_w1_skill = 0.5 #prior
        expected_w2_skill = 0.5

#       print("test dai difficulties known: d_res = " + str(d_res['posteriors']))
#       print "D_RES", d_res
#       raw_input()

        self.assertAlmostEqual(expected_q1_post, d_res['posteriors'][1])
        self.assertAlmostEqual(expected_q2_post, d_res['posteriors'][2])

        self.assertAlmostEqual(expected_w1_skill, d_res['workers'][1])
        self.assertAlmostEqual(expected_w2_skill, d_res['workers'][2])

        print("Test Dai w/ Difficulties Known, Skills Unknown:")
        print(d_res)

    def dai_helper(self, votes, workers, questions):
        d = InferenceModule(method = 'dai')
        d_res = d.estimate(votes, workers, questions)
        return d_res

#   def test_mdp(self):
#       # test mdp
#       m = InferenceModule(method = 'mdp')
#       m_res = m.estimate(self.votes, self.workers, self.questions)
#       print m_res
        
if __name__ == '__main__':
    unittest.main()
