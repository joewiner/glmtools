""" Run GLM on final dimension of 4D arrays
"""

import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
# Print array values to 4 decimal places
np.set_printoptions(precision=4)
import scipy.stats
from glm import glm, t_test


def glm_4d(Y, X):
    """ Run GLM on on 4D data `Y` and design `X`

    Parameters
    ----------
    Y : array shape (I, J, K, T)
        4D array to fit to model with design `X`.  Column vectors are vectors
        over the final length T dimension.
    X : array ahape (T, P)
        2D design matrix to fit to data `Y`.

    Returns
    -------
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.
    """
    #reshape for glm function
    Y_2D = Y.reshape(-1, Y.shape[-1]).T
    #plug in to glm
    B, sigma_2, df = glm(Y_2D, X)
    #reshape B and sigma_2
    B_4D = B.reshape(Y.shape[0],Y.shape[1],Y.shape[2],X.shape[1])
    sigma_2_3D = sigma_2.reshape(Y.shape[0],Y.shape[1],Y.shape[2])

    return B_4D, sigma_2_3D, df


def t_test_3d(c, X, B, sigma_2, df):
    """ Two-tailed t-test on 3D estimates given contrast `c`, design `X`

    Parameters
    ----------
    c : array shape (P,)
        contrast specifying conbination of parameters to test.
    X : array shape (N, P)
        design matrix.
    B : array shape (I, J, K, P)
        parameter array, one length P vector of parameters for each voxel.
    sigma_2 : array shape (I, J, K)
        unbiased estimate of variance for each voxel.
    df : int
        degrees of freedom due to error.

    Returns
    -------
    t : array shape (I, J, K)
        t statistics for each data vector.
    p : array shape (V,)
        two-tailed probability value for each t statistic.
    """
    #reshape for t-test loop
    B_2D = B.reshape(-1, B.shape[-1]).T
    sigma_2_1D = sigma_2.reshape(np.prod(sigma_2.shape))
    #loop through every voxel
    #for i in range(np.prod(sigma_2.shape)):
    #    t, p = t_test(c, X[:,i], B_2D[:,i], sigma_2[i], df)
    #    t_array[i] = t
    #    p_vec[i] = p
    t, p = t_test(c, X, B_2D, sigma_2_1D, df)
    #reshape
    t_3D = t.reshape(Y.shape[0],Y.shape[1],Y.shape[2])
    p_3D = p.reshape(Y.shape[0],Y.shape[1],Y.shape[2])

    return t_3D, p_3D

def test_glm4d_t_test3d():
    #generate fake data to test functions
    n = 20
    x = np.random.normal(10, 2, size=n**3)
    X = np.ones((n**3, n))
    X[:, 1] = x
    X = X.T
    Y = np.random.normal(20, 1, size=(n,n,n,n))

    B, sigma_2, df = glm_4d(Y,X)
    c = np.ones(n**3)
    t, p = t_test_3d(c, X, B, sigma_2, df)

    res = scipy.stats.linregress(x, Y)

    return np.allclose(p, res.pvalue)
