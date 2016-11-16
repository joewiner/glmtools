""" py.test test for glmtools code

Run with:

    py.test glmtools
"""

import numpy as np
from glmtools import glm, t_test
from numpy.testing import assert_almost_equal
import scipy.stats

def test_glm_t_test():
    """ needs to be tested with Y matrix that has 2 groups of 10
    """
    #generate fake data to test functions
    n = 20
    x = np.random.normal(10, 2, size=n)
    X = np.ones((n, 2))
    X[:, 1] = x
    #Y = np.random.normal(20, 1, size=n)

    #Try doing something like:
    Y = np.random.normal(20, 4, size=n)
    #then, to test against scipy, loop through the columns
    #checking the parameters, p values, t value.

    B, sigma_2, df = glm(Y,X)
    c = np.array([0, 1])
    t, p = t_test(c, X, B, sigma_2, df)

    res = scipy.stats.linregress(x, Y)

    #return np.allclose(p, res.pvalue)
    assert_almost_equal(res.pvalue, p)
    return
