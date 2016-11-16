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
    B = npl.pinv(X).dot(Y)
    E = Y - X.dot(B)

    df = len(X) - npl.matrix_rank(X)

    E_sq = E ** 2
    sigma_2 = np.sum(E_sq, axis=0) / df

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
    #test allows for multiple columns of y
    b_cov = npl.pinv(X.T.dot(X))
    t = c.dot(B) / np.sqrt(sigma_2 * c.dot(b_cov).dot(c))

    #get two-tailed p-value for t statistic
    t_dist = scipy.stats.t(df=df)

    p = 2 * (1 - t_dist.cdf(np.abs(t)))
    return t, p
