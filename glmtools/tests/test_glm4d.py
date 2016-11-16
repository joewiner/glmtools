""" Test glm4d module

Run with:

    py.test glmtools
"""

import numpy as np
from glmtools import glm_4d, t_test_3d, glm, t_test
from numpy.testing import assert_almost_equal, assert_equal
import scipy.stats


def test_glm4d():
    #generate fake data to test functions
    n = 5
    x = np.random.normal(5, 1, size=(n, n**3-1))
    X = np.ones((n, n**3))
    X[:, 1:] = x

    Y = np.random.normal(5, 1, size=(n,n,n,n))

    B, sigma_2, df = glm_4d(Y,X)
    c = np.array([2, 3, 2, 3, 2])
    t, p = t_test_3d(c, X, B, sigma_2, df)

    res = scipy.stats.linregress(x, Y.ravel())

    assert_almost_equal(res.pvalue, p)
    return
