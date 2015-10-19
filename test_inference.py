import unittest
from inference import *

class InferenceTest(unittest.TestCase):
    def setUp(self):
        self.votes = {(345, 2): {'vote': 0},
                      (1, 2): {'vote': 1},
                      (1, 3): {'vote': 1}}
                      
        self.workers = {345: {'skill': 0.7},
                        1: {'skill': 0.6}}
        
        self.questions= {2: {'difficulty': None},
                         3: {'difficulty': None}}
    
    def test_infer(self):
        """BUG: Incomplete."""
        # test dai
        d = InferenceModule(method = 'dai')
        d.estimate(self.votes, self.workers, self.questions)
        
        # test mdp
        m = InferenceModule(method = 'mdp')
        m.estimate(self.votes, self.workers, self.questions)
        
if __name__ == '__main__':
    unittest.main()
