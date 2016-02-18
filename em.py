"""
Difference from jbragg Inference module:
    skills are not inverted
"""
# TODO use util functions for calculating probabilities

import numpy as np
import scipy.optimize
from scipy.stats import pearsonr
from scipy.special import gamma

SKILL_BOUND = (0.01, None)
DIFF_BOUND = (0.01, 0.99)
USE_DIFFICULTY_PRIOR = False

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
#   print "dbeta: x=",x
    assert (x > 0).all()
    assert (x < 1).all()

    return gamma(a+b)/(gamma(a)*gamma(b)) * \
           ((a-1) * x**(a-2) * (1-x)**(b-1) - x**(a-1) * (b-1) * (1-x)**(b-2))

def func(posteriors, B, D, S):
    """
    Computes E[ln P(v,b|d,y)]
    """

    # TERM 1: Sum(t)Sum(a=0,1) P(vt = a|b,d,y) * ln P(vt = a)
    t1 = 0

    label_prob = 0.5
    t1 += np.sum(posteriors)*np.log(label_prob)
    t1 += np.sum(1-posteriors)*np.log(1-label_prob)

    # TERM 2: Sum(w,k)Sum(a=0,1) P(vt = a|b,d,y) * ln P(b_t,w,k | vt = a,
    # d_kt, y_w)
    probs = 0.5 * (1 + (1 - D)**S[:, np.newaxis])

    true_votes = (B == 1)
    false_votes = (B == 0)

    ptrue = \
        np.sum(np.log(probs) * true_votes, 0) + \
        np.sum(np.log(1 - probs) * false_votes, 0)

    pfalse = \
        np.sum(np.log(probs) * false_votes, 0) + \
        np.sum(np.log(1 - probs) * true_votes, 0)

    t2 = np.sum(posteriors * ptrue + (1 - posteriors) * pfalse)

    return t1 + t2

def CALC_DS(posteriors, B, D, x):

    true_votes = (B == 1)
    false_votes = (B == 0)

    probs = 0.5 * (1 + (1 - D)**x[:, np.newaxis])
    probs_ds = 0.5 * (1 - D)**x[:, np.newaxis] * np.log(1 - D)

    ptrue_ds = \
                1 / probs * probs_ds * true_votes + \
                1 / (1 - probs) * (-probs_ds) * false_votes

    pfalse_ds = \
                1 / probs * probs_ds * false_votes + \
                1 / (1 - probs) * (-probs_ds) * true_votes

    ds = np.sum(posteriors * ptrue_ds +
                (1 - posteriors) * pfalse_ds, 1)

    return ds


def CALC_DD(posteriors, B, x, S):

    true_votes = (B == 1)
    false_votes = (B == 0)

    probs = 0.5 * (1 + (1 - x)**S[:, np.newaxis])

    probs_dd = -0.5 * S[:, np.newaxis] * (1 - x)**((S - 1)[:, np.newaxis])

    ptrue_dd = \
            np.sum(1/probs*probs_dd * true_votes, 0) + \
            np.sum(1/(1-probs)*(-probs_dd) * false_votes, 0)

    pfalse_dd = \
            np.sum(1/probs*probs_dd * false_votes, 0) + \
            np.sum(1/(1-probs)*(-probs_dd) * true_votes, 0)
    dd = posteriors * ptrue_dd + \
            (1-posteriors) * pfalse_dd

    return dd


def EM(observations, nq, nw, spec_bounds, diffs=None, skills=None):
    def E(params):
        D = params[:nq]
        S = params[nq:]

        prior = 0.5
        priors = prior * np.ones(nq)
        probs = 0.5 * (1 + np.power((1 - D), S[:, np.newaxis]))

        true_votes = (observations == 1)
        false_votes = (observations == 0)

        # log P(U = true,  votes)
        ptrue = np.log(priors) + np.sum(np.log(probs) * true_votes, 0) + \
            np.sum(np.log(1 - probs) * false_votes, 0)
        # log P(U = false,  votes)
        pfalse = np.log(1 - priors) + np.sum(np.log(probs) * false_votes, 0) + \
            np.sum(np.log(1 - probs) * true_votes, 0)

        # log P(votes)
        norm = np.logaddexp(ptrue, pfalse)

        #posteriors, ll
        return np.exp(ptrue-norm), np.sum(norm)

    def M(posteriors, params):
        # print "M: posteriors, params =", posteriors, params
        # print params
        D = params[:nq]
        S = params[nq:]

        def f(x):
            curD = x[:nq]
            curS = x[nq:]

            v = func(posteriors, observations, curD, curS)
            dd = CALC_DD(posteriors, observations, curD, curS)
            ds = CALC_DS(posteriors, observations, curD, curS)

            # include prior on difficulty
            if not knownD and USE_DIFFICULTY_PRIOR:
                v += np.sum(np.log(scipy.stats.beta.pdf(curD,1.01,1.01)))
                pr = scipy.stats.beta.pdf(curD,1.01,1.01)
                dd += 1/pr * dbeta(curD,1.01,1.01)
            
            # TODO sloppy
            if knownD and not knownS:
                dd = np.zeros(nq)
            elif knownS and not knownD:
                ds = np.zeros(nw)
            jac = np.hstack((dd, ds))
            return (-v, -jac)

        # should be able to start anywhere in bounds (M-step opt is convex)
        init = params
#       init = [0.5 for i in range(len(D)+len(S))]

        res = scipy.optimize.minimize(
            f,
            init,
            method='L-BFGS-B',
            jac=True,
            bounds=spec_bounds,
            options={'disp': False})
        if not res.success:
            print res
        assert res.success

        return res.x

    # Specify previously known difficulties, skills
    if diffs and not None in diffs:
        knownD = True
        init_diffs = diffs
    else:
        knownD = False
        init_diffs = np.array([0.1 + 0.8 * np.random.random()
                               for i in range(nq)])

    if skills and not None in skills:
        knownS = True
        init_skills = skills
    else:
        knownS = False
        init_skills = np.array([0.1 + 0.8 * np.random.random()
                                for i in range(nw)])
    params = np.concatenate([init_diffs, init_skills], 0)

    if knownD and knownS:
        # parameters known; just compute posteriors in E-step
        posteriors, _ = E(params)
        return {'observations': observations, 'difficulties': diffs, 'skills': skills}, posteriors

    ll = float('-inf')
    ll_change = float('inf')
    em_round = 0
    while ll_change > 0.001:
        #       print "EM round: " + str(em_round)
        #       print "params:",params
        posteriors, ll_new = E(params)
        newParams = M(posteriors, params)

        if ll == float('-inf'):
            ll_change = float('inf')
        else:
            ll_change = (ll_new - ll) / np.abs(ll)  # percent increase

        if ll_change < 0.0:
            #           print "em_round", em_round
            #           print "posteriors", posteriors
            #           print "params", params
            #           print "newParams", newParams
            #           print "ll", ll
            #           print "ll_new", ll_new
            #           print "ll_change", ll_change
            pass
        else:
            # only update parameters if the last M-step was an improvement
            params = newParams

        # log likelihood increases monotonically!
        # TODO XXX problem: this assert could fail on the last round of EM
#       assert ll_change >= 0.0
        ll = ll_new
        em_round += 1
#       print "after round", em_round
#       print posteriors
#       print params
#       print ll_new
#       print ll_change

    outParams = {'observations': observations,
                 'difficulties': params[:nq], 'skills': params[nq:]}
    return outParams, posteriors


def estimate(votes, workers, questions):
    # TODO figure out best way to handle 0 observations
    if len(votes) == 0:
        posteriors = {}
        diffs = {}
        skills = {}
        for question in questions:
            posteriors[question] = 0.5
            diffs[question] = 0.5
        for worker in workers:
            skills[worker] = 0.5

        return {'posteriors': posteriors, 'questions': diffs, 'workers': skills}

    # assuming skill+diff unknown for now
    skills = []
    diffs = []

    w_id_to_idx = {}
    nw = 0
    for worker in workers:
        skills.append(workers[worker]['skill'])
        w_id_to_idx[worker] = nw
        nw += 1
    q_id_to_idx = {}
    nq = 0
    for question in questions:
        diffs.append(questions[question]['difficulty'])
        q_id_to_idx[question] = nq
        nq += 1

    observations = -1 * np.ones(shape=(nw, nq))
    for vote in votes:
        w_id, q_id = vote
        w_idx = w_id_to_idx[w_id]
        q_idx = q_id_to_idx[q_id]
        observations[w_idx][q_idx] = votes[vote]['vote']

#   nw = len(workers)
#   nq = len(questions)
    bounds = [DIFF_BOUND for i in range(nq)] + [SKILL_BOUND for i in range(nw)]
    params, posteriors_array = EM(observations, nq, nw, bounds, diffs, skills)

    # convert ids back
    posteriors = {}
    diffs = {}
    skills = {}
    for question in questions:
        q_idx = q_id_to_idx[question]
        posteriors[question] = posteriors_array[q_idx]
        diffs[question] = params['difficulties'][q_idx]
    for worker in workers:
        w_idx = w_id_to_idx[worker]
        skills[worker] = params['skills'][w_idx]

    return {'posteriors': posteriors, 'questions': diffs, 'workers': skills}
