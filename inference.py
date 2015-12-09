from __future__ import division
import numpy as np
import scipy
import scipy.stats
import scipy.misc
from scipy.special import gamma
from collections import defaultdict
import prob

#---- Helpers
def dbeta(x, a, b):
    """Beta derivative.

    >>> round(dbeta(0.5, 2, 2), 10)
    0.0
    >>> round(dbeta(0.6, 2, 2), 10)
    -1.2
    >>> round(dbeta(0.9, 1, 1), 10)
    0.0

    """
    x = np.array(x)
    #http://www.math.uah.edu/stat/special/Beta.html
    #B(a,b)=Gamma(a)*Gamma(b)/Gamma(a+b)
    #derivative of beta distribution
    #f'(x) = (1/B(a,b)) * x^(a-2) * (1-x)^(b-2) * [(a-1)-(a+b-2)*x], 0<x<1
    assert (x > 0).all()
    assert (x < 1).all()

    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

#----- Main

class InferenceModule():
    def __init__(self, method = 'dai'):
        """Initialize as 'dai' or 'mdp' module"""
        self.method = method
        
        if method == 'dai':
            ProbModule = prob.ProbabilityDai()
            self.bounds = {'difficulty': (0.000000001,0.9999999999),
                           'skill': (0.000000001,None)}
            self.em_init_params = {'difficulty': 0.5,
                                   'skill': 0.5,
                                   'label':0.5}
        elif method == 'mdp':
            ProbModule = prob.ProbabilityMDP()
            self.bounds = {'difficulty': (0.000000001,0.9999999999),
                           'skill': (0.500000001,0.9999999999)}
            self.em_init_params = {'difficulty': 0.5,
                                   'skill': 0.75,
                                   'label':0.5}
        
        self.allprobs = ProbModule.allprobs
        self.allprobs_ddifficulty = ProbModule.allprobs_ddifficulty
        self.allprobs_dskill = ProbModule.allprobs_dskill
        
        # BUG: this prior is for difficulty only
        # BUG: has different effect on [0.5,1] (MDP) and [0,1] (Dai)
        self.prior = (1.01,1.01)

    def estimate(self, votes, workers, questions, buckets=False):
        """

        Args:
            votes:      {(worker_id, question_id): {'vote': 0/1}}
            workers:    {worker_id: {'skill': 0.7}}
            questions:  {question_id: {'difficulty': 0.6}}

        """

        sorted_q = sorted(questions)
        q_id = dict((i_q, i) for i,i_q in enumerate(sorted_q))            
        self.gt_difficulties = [questions[i]['difficulty'] for i in sorted_q]
        
        sorted_w = sorted(workers)
        w_id = dict((i_w, i) for i,i_w in enumerate(sorted_w))
        self.gt_skills = [workers[i]['skill'] for i in sorted_w]

        # Assume all difficulties/skills known/unknown.
        self.known_difficulty = not None in self.gt_difficulties
        self.known_skill = not None in self.gt_skills
        
        self.num_workers = len(sorted_w)
        self.num_questions = len(sorted_q)
        self.init_observations()
        for w,q in votes:
            self.observations[w_id[w], q_id[q]] = votes[w,q]['vote']
        
        # Run difficulty buckets version.
        if buckets and not self.known_difficulty:
            marginals, _ = self.infer_difficulty_buckets(self.observations)
            # Let's just return MAP difficulty estimate for now.
            difficulties = np.sum([i * marginals['difficulty'][i] for
                                   i in marginals['difficulty']], 0)
            response = {'posteriors': dict((q_id, marginals['answer'][i]) for
                                            i,q_id in enumerate(sorted_q)),
                        'questions': dict((q_id, difficulties[i]) for
                                          i,q_id in enumerate(sorted_q))}
        else: # Run old version.
            params, posteriors = self.run_em()
        
            response = {'posteriors': dict((q_id, posteriors[i]) for
                                           i,q_id in enumerate(sorted_q)),
                        'questions': dict((q_id, params['difficulties'][i]) for
                                           i,q_id in enumerate(sorted_q)),
                        'workers': dict((w_id, params['skills'][i]) for
                                        i,w_id in enumerate(sorted_w)),
                        'label': params['label']}
        return response
                                        
    #----------------------
    def init_params(self):
        if self.known_skill and self.known_difficulty:
            params = {'difficulties': self.gt_difficulties,
                      'skills': self.gt_skills,
                      'label': self.em_init_params['label']}
        elif self.known_skill:
            params = {'difficulties': np.ones(self.num_questions) * \
                                       self.em_init_params['difficulty'],
                      'skills': self.gt_skills,
                      'label': self.em_init_params['label']}
        else:
            # params = {'difficulties':np.random.random(self.num_questions),
            #           'skills':np.random.random(self.num_workers),
            #           #'label':np.random.random()}
            #           'label':0.5}
            params = {'difficulties': np.ones(self.num_questions) * \
                                       self.em_init_params['difficulty'],
                      'skills': np.ones(self.num_workers) * \
                                self.em_init_params['skill'],
                      'label': self.em_init_params['label']}
                       
        return params
         
     
    def init_observations(self):
        """observations is |workers| x |questions| matrix
        -1 - unobserved
        1 - True
        0 - False

        """
        self.observations = np.zeros((self.num_workers, self.num_questions))-1
        return
        
        
    def run_em(self):
        """Learn params and posteriors"""
        observations = self.observations
        known_s = self.known_skill
        known_d = self.known_difficulty
        

        def E(params):
            post, ll = self.infer(observations, params)
        
            if not known_d:
            # add prior for difficulty (none for skill)
                ll += np.sum(np.log(scipy.stats.beta.pdf(
                                        params['difficulties'],
                                        self.prior[0],
                                        self.prior[1])))
            
                                  
            # add beta prior for label parameter
            #ll += np.sum(np.log(scipy.stats.beta.pdf(params['label'],
            #                                     self.prior[0],
            #                                     self.prior[1])))
                
            
            return post, ll / self.num_questions
    
        def M(posteriors, params_in):
            params = dict()
            #params['label'] = (self.prior[0] - 1 + sum(posteriors)) / \
            #                  (self.prior[0] - 1 + self.prior[1] - 1 + \
            #                   self.num_questions)
            params['label'] = 0.5 # hard-code for this exp


            def f(params_array):
                if not known_d and known_s:
                    difficulties = params_array
                    skills = self.gt_skills
                elif not known_s and known_d:
                    skills = params_array
                    difficulties = self.gt_difficulties
                else:  # both skill and difficulty unknown
                    difficulties = params_array[:self.num_questions]
                    skills = params_array[self.num_questions:]



                probs = self.allprobs(skills,
                                      difficulties)
                probs_dd = self.allprobs_ddifficulty(skills,
                                                     difficulties)
                probs_ds = self.allprobs_dskill(skills,
                                                difficulties)
                              


#                    priors = prior * np.ones(self.num_questions)

                true_votes = (observations == 1)   
                false_votes = (observations == 0)   



#                    ptrue = np.log(priors) + \
                ptrue = \
                        np.sum(np.log(probs) * true_votes, 0) + \
                        np.sum(np.log(1-probs) * false_votes, 0)
#                    pfalse = np.log(1-priors) + \
                pfalse = \
                         np.sum(np.log(probs) * false_votes, 0) + \
                         np.sum(np.log(1-probs) * true_votes, 0)

                ptrue_dd = \
                        np.sum(1/probs*probs_dd * true_votes, 0) + \
                        np.sum(1/(1-probs)*(-probs_dd) * false_votes, 0)

                pfalse_dd = \
                        np.sum(1/probs*probs_dd * false_votes, 0) + \
                        np.sum(1/(1-probs)*(-probs_dd) * true_votes, 0)

                ptrue_ds = \
                        1/probs*probs_ds * true_votes + \
                        1/(1-probs)*(-probs_ds) * false_votes

                pfalse_ds = \
                        1/probs*probs_ds * false_votes + \
                        1/(1-probs)*(-probs_ds) * true_votes

                # print '--------------'
                # print skills
                # print difficulties
                # print probs
                # print probs_dd
                # print true_votes
                # print ptrue_dd
                # print false_votes
                # print pfalse_dd
                # print posteriors

                # result
                v = np.sum(posteriors * ptrue + (1-posteriors) * pfalse)
                dd = np.array(posteriors * ptrue_dd + \
                              (1-posteriors) * pfalse_dd)
                ds = np.sum(posteriors * ptrue_ds + \
                              (1-posteriors) * pfalse_ds, 1)


                #dd = np.append(dd, np.sum(posteriors * 1/priors + \
                #                          (1-posteriors) * -1/(1-priors)))

#                    print '---'
#                    print params_array
#                    print -v
#                    print
#                    print
#                    print
#                print dd
#                    print '---'

                pr = scipy.stats.beta.pdf(difficulties,*self.prior)

                #                    print '************jjjjjj'
                v += np.sum(np.log(pr))
                dd += 1/pr * dbeta(difficulties,*self.prior)
                #print difficulties, -v, -dd

                if not known_d and known_s:
                    jac = dd 
                elif not known_s and known_d:
                    jac = ds
                else: 
                    jac = np.hstack((dd,ds))


                # return negative to minimizer
                return (-v,
                        -jac)
                #                    return -v


            # init_d = 0.1 * np.ones(self.num_questions)
            init_d = params_in['difficulties']
            bounds_d = [self.bounds['difficulty'] for 
                        i in xrange(self.num_questions)]
#                init_s = 0.9 * np.ones(self.num_workers)
            init_s = params_in['skills']
            bounds_s = [self.bounds['skill'] for 
                        i in xrange(self.num_workers)]

            if not known_d and known_s:
                init = init_d
                bounds = bounds_d
            elif not known_s and known_d:
                init = init_s
                bounds = bounds_s
            else: 
                init = np.hstack((init_d,init_s))
                bounds = bounds_d + bounds_s

            res = scipy.optimize.minimize(
                        f,
                        init,
                        method='SLSQP',#TODO: understand why minimization fails with L-BFGS-B
                        jac=True,
                        bounds=bounds,
                        options={'disp':False})
#                print res.x
            assert res.success
            # print 'success: ',res.success
            if not known_d and known_s:
                params['difficulties'] = res.x
                params['skills'] = self.gt_skills
            elif not known_s and known_d:
                params['skills'] = res.x
                params['difficulties'] = self.gt_difficulties
            else: 
                params['difficulties'] = res.x[:self.num_questions]
                params['skills'] = res.x[self.num_questions:]


#                print params['skills']
#                print params['difficulties']
            return params
#                return {'label': res.x[self.num_questions],
#                        'difficulties': res.x[0:self.num_questions]}

        params = self.init_params()

        # if known parameters, just run E step
        if known_s and known_d:
            posteriors, _ = E(params)
            return params, posteriors

        # otherwise, run EM
        ll = float('-inf')
        ll_change = float('inf')
        em_round = 0
        while ll_change > 0.001:  # run while ll increase is at least .1%
#            print 'EM round: ' + str(em_round)
            posteriors,ll_new = E(params)
            params = M(posteriors, params)

            if ll == float('-inf'):
                ll_change = float('inf')
            else:
                ll_change = (ll_new - ll) / np.abs(ll) # percent increase

            ll = ll_new
            # print 'em_round: ' + str(em_round)
            # print 'll_change: ' + str(ll_change)
            # print 'log likelihood: ' + str(ll)
            # print 'skills ', params['skills'][:5]
            # print 'diffs ', params['difficulties'][:5]
            # print 'posteriors ', posteriors
            # print

            # NOTE: good to have this assert, but fails w/ gradient ascent
            #assert ll_change > -0.001  # ensure ll is nondecreasing

            em_round += 1

#        print str(em_round) + " EM rounds"
#        print params['label']
#        print params['difficulties']
        return params, posteriors


    def infer(self, observations, params):
        """Probabilistic inference for question posteriors.
        
        Observation matrix has been observed.

        """

        prior = params['label']
        probs = self.allprobs(params['skills'], params['difficulties'])
        priors = prior * np.ones(self.num_questions)
     
        true_votes = (observations == 1)   
        false_votes = (observations == 0)   

        # log P(U = true,  votes)
        ptrue = np.log(priors) + np.sum(np.log(probs) * true_votes, 0) + \
                                 np.sum(np.log(1-probs) * false_votes, 0)
        # log P(U = false,  votes)
        pfalse = np.log(1-priors) + np.sum(np.log(probs) * false_votes, 0) + \
                                    np.sum(np.log(1-probs) * true_votes, 0)

        # log P(votes)
        norm = np.logaddexp(ptrue, pfalse)

        return np.exp(ptrue) / np.exp(norm), np.sum(norm)
    
    def infer_difficulty_buckets(self, observations):
        """Hack to infer difficulty and true answers."""
        params = self.init_params()
        prior = params['label']
        
        num_questions = np.size(observations, 1)
        num_buckets = 11
        buckets = np.linspace(0, 1, num_buckets)
        
        # BUG: hard-code equal probability difficulties for now (matches experiments)
        params['difficulty'] = dict((p,1/num_buckets) for p in buckets)
        
        probs = dict()
        for i in params['difficulty']:
            difficulties = np.ones(num_questions) * i
            probs[i] = self.allprobs(params['skills'], difficulties)
        priors = prior * np.ones(num_questions)
     
        true_votes = (observations == 1)   
        false_votes = (observations == 0)
                
        joint = dict()
        for i in probs:
            # log P(U = true, D = diff, votes)
            ptrue = np.log(priors) + np.log(params['difficulty'][i]) + \
                                     np.sum(np.log(probs[i]) * true_votes, 0) + \
                                     np.sum(np.log(1-probs[i]) * false_votes, 0)
            # log P(U = false, D = diff, votes)
            pfalse = np.log(1-priors) + np.log(params['difficulty'][i]) + \
                                        np.sum(np.log(probs[i]) * false_votes, 0) + \
                                        np.sum(np.log(1-probs[i]) * true_votes, 0)
            
            joint[True, i] = ptrue
            joint[False, i] = pfalse

        # log P(votes)
        norm = scipy.misc.logsumexp(joint.values(), 0)
        
        #----- compute marginals
        # P(U = true, D = diff | votes)
        posteriors = dict()
        for k in joint:
            posteriors[k] = joint[k] - norm
        
        marginals = dict()
        
        # P(U = true | votes)
        marginals['answer'] = np.exp(scipy.misc.logsumexp([posteriors[k] for
                                                           k in posteriors if k[0]], 0))
        
        # P(diffficulty = d | votes)
        marginals['difficulty'] = dict()                                                    
        for i in set(k[1] for k in posteriors):
            marginals['difficulty'][i] = np.exp(np.logaddexp(posteriors[True, i],
                                                             posteriors[False, i]))
        # just return marginals, not posteriors
        return marginals, np.sum(norm)

        
class InferenceTest():
    def __init__(self):
        """
        votes = {(worker_id, question_id): {'vote': 0/1}}
        workers = {worker_id: {'skill': 0.7}}
        questions = {question_id: {'difficulty': 0.6}}
        """
        self.votes = {(345, 2): {'vote': 0},
                      (1, 2): {'vote': 1},
                      (1, 3): {'vote': 1}}
                      
        self.workers = {345: {'skill': 0.7},
                        1: {'skill': 0.6}}
        
        self.questions={2: {'difficulty': None},
                        3: {'difficulty': None}}
    
    def test(self):
        # test dai
        d = InferenceModule(method = 'dai')
        d.estimate(self.votes, self.workers, self.questions)
        
        # test mdp
        m = InferenceModule(method = 'mdp')
        m.estimate(self.votes, self.workers, self.questions)
