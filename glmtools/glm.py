""" Functions for running GLM on 2D and 3D data
"""

import numpy as np
import numpy.linalg as npl
import scipy.stats

def glm(Y, X):
    """ Run GLM on on data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (N, V)
        1D or 2D array to fit to model with design `X`.  `Y` is column
        concatenation of V data vectors.
    X : array ahape (N, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (P, V)
        parameter matrix, one column for each column in `Y`.
    sigma_2 : array shape (V,)
        unbiased estimate of variance for each column of `Y`.
    df : int
        degrees of freedom due to error.
    """
    N = len(X)
    B = npl.pinv(X).dot(Y)
    E = Y - X.dot(B)

    df = N - npl.matrix_rank(X)
    sigma_2 = np.sum(E ** 2) / df

    return B, sigma_2, df

def t_test(c, X, B, sigma_2, df):
    """ Two-tailed t-test given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (P, V)
        parameter estimates for V vectors of data.
    sigma_2 : float
        estimate for residual variance.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (V,)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """

    #get t statistic
    c_b_cov = c.dot(npl.pinv(X.T.dot(X))).dot(c)
    t = c.dot(B) / np.sqrt(sigma_2 * c_b_cov)

    #get two-tailed p-value for t statistic
    t_dist = scipy.stats.t(df=df)
    p = 2*(1 - t_dist.cdf(t))
    return t, p

def test_glm_t_test(Y):
    """ needs to be tested with Y matrix that has 2 groups of 10
    """
    res = scipy.stats.ttest_ind(Y[:10], Y[10:])
    glm(Y,X)
    t,p = t_test(c, X, B, sigma_2, df)
