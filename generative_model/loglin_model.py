import numpy as np
import numba
from numba import jit

def mle_2x2_ind_old(cont_tab):
    """
    Computes the maximum likelihood estimates of a 2X2 table under the hypothesis of independence.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of independence
    -------
    """
    #Computes the chisquare statistics and its p-value for a 2X2X2 contingency table under the total independence hypothesis

    row_sums = np.sum(cont_tab, axis=1)
    col_sums = np.sum(cont_tab, axis=0)
    n = np.sum(cont_tab)

    expected = np.random.rand(2, 2)
    for i in range(2):
        for j in range(2):
            expected[i, j] = row_sums[i] * col_sums[j] / n

    return expected

@jit(nopython=True)
def mle_2x2_ind(cont_tab):
    """
    Computes the maximum likelihood estimates of a 2X2 table under the hypothesis of independence.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of independence
    -------
    """
    #Computes the chisquare statistics and its p-value for a 2X2X2 contingency table under the total independence hypothesis

    row_sums = np.sum(cont_tab, axis=1)
    col_sums = np.sum(cont_tab, axis=0)
    n = np.sum(cont_tab)

    expected = np.random.rand(2, 2)
    for i in range(2):
        for j in range(2):
            expected[i, j] = row_sums[i] * col_sums[j] / n

    return expected


def mle_2x2_ind_vector(cont_tables, n):
    """
    Computes the maximum likelihood estimates of a 2X2 table under the hypothesis of independence.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of independence
    -------
    """
    # Computes the chisquare statistics and its p-value for a 2X2X2 contingency table under the total independence hypothesis

    row_sums = np.sum(cont_tables, axis=1, keepdims=True)
    col_sums = np.sum(cont_tables, axis=2, keepdims=True)

    return np.matmul(col_sums, row_sums) / n

def mle_2x2x2_ind(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis of independence.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of independence
    -------
    """
    #Computes the chisquare statistics and its p-value for a 2X2X2 contingency table under the total independence hypothesis

    m__k = np.sum(np.sum(cont_cube, axis=2), axis=1)
    m_j_ = np.sum(np.sum(cont_cube, axis=1), axis=0)
    mi__ = np.sum(np.sum(cont_cube, axis=2), axis=0)
    n = np.sum(cont_cube)

    row_props = mi__/n
    col_props = m_j_/n
    depth_props = m__k/n
    expected = np.random.rand(2,2,2)
    for k in range(2):
        for i in range(2):
            for j in range(2):

                expected[k,i,j] = row_props[i]*col_props[j]*depth_props[k]*n

    return expected

def mle_2x2x2_AB_C(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis Variable 3 is independent of variable
    1 and 2 together.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of Variable 3 is independent of variable
    1 and 2 together.
    -------
    """

    m__k = np.sum(np.sum(cont_cube, axis=2), axis=1)
    mij_ = np.sum(cont_cube, axis=0)
    n = np.sum(cont_cube)

    expected = np.random.rand(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):

                expected[k,i,j] = mij_[i,j]*m__k[k]/n

    return expected

def mle_2x2x2_AC_B(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis Variable 2 is independent of variable
    1 and 3 together.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of Variable 2 is independent of variable
    1 and 3 together.
    -------
    """
    m_j_ = np.sum(np.sum(cont_cube, axis=1), axis=0)
    mi_k = np.sum(cont_cube, axis=2).T
    n = np.sum(cont_cube)

    df = 1 # TODO df = 1???

    expected = np.random.rand(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):

                expected[k,i,j] = mi_k[i,k]*m_j_[j]/n

    return expected

def mle_2x2x2_BC_A(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis Variable 1 is independent of variable
    2 and 3 together.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of Variable 1 is independent of variable
    2 and 3 together.
    -------
    """
    mi__ = np.sum(np.sum(cont_cube, axis=0), axis=1)
    m_jk = np.sum(cont_cube, axis=1).T
    n = np.sum(cont_cube)

    df = 1 # TODO df = 1???

    expected = np.random.rand(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):

                expected[k,i,j] = m_jk[j,k]*mi__[i]/n

    return expected

def mle_2x2x2_AC_BC(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis Variable 1 and 2 are indepndent in
     each layer of variable 3.
    1 and 2 together.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of Variable 1 and 2 are indepndent in
     each layer of variable 3.
    -------
    """

    m_jk = np.sum(cont_cube, axis=1).T
    mi_k = np.sum(cont_cube, axis=2).T
    m__k = np.sum(np.sum(cont_cube, axis=2), axis=1)

    expected = np.random.rand(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):

                expected[k,i,j] = m_jk[j,k]*mi_k[i,k]/m__k[k]

    return expected

def mle_2x2x2_AB_BC(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis Variable 1 and 3 are indepndent in
     each layer of variable 2.
    1 and 2 together.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of Variable 1 and 3 are indepndent in
     each layer of variable 2.
    -------
    """

    mij_ = np.sum(cont_cube, axis=0)
    m_jk = np.sum(cont_cube, axis=1).T
    m_j_ = np.sum(np.sum(cont_cube, axis=1), axis=0)

    expected = np.random.rand(2,2,2)
    for i in range(2):
        for j in range(2):
            for k in range(2):

                expected[k,i,j] = m_jk[j,k]*mij_[i,j]/m_j_[j]

    return expected


def mle_2x2x2_AB_AC(cont_cube):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis Variable 2 and 3 are indepndent in
     each layer of variable 1.
    1 and 2 together.
    The formula for the MLEs are obtained via the method proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3. The formulas are summarized in Birch (1963) Maximum Likelihood in Three-Wat Contingency Tables
    Parameters
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of Variable 2 and 3 are indepndent in
     each layer of variable 1.
    -------
    """

    mij_ = np.sum(cont_cube, axis=0)
    mi_k = np.sum(cont_cube, axis=2).T
    mi__ = np.sum(np.sum(cont_cube, axis=0), axis=1)

    expected = np.random.rand(2, 2, 2)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                expected[k, i, j] = mi_k[i,k] * mij_[i, j] / mi__[i]

    return expected

def iterative_proportional_fitting_AB_AC_BC(cont_cube, delta=0.01):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis no three-factor interaction.
    1 and 2 together.
    The formula for the MLEs are obtained via the iterative proportional fitting proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3.
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    delta (float) : If the difference between the last estimates and the current estimates, we accept the estimates.
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of no three-factor interaction
    -------
    """

    xij_ = np.sum(cont_cube, axis=0)
    xi_k = np.sum(cont_cube, axis=2).T
    x_jk = np.sum(cont_cube, axis=1).T

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)
        mij_ = np.sum(mijk, axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xij_[i,j] / mij_[i, j]

        mi_k = np.sum(mijk, axis=2).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xi_k[i,k] / mi_k[i, k]

        m_jk = np.sum(mijk, axis=1).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * x_jk[j,k] / m_jk[j, k]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk
#@jit(nopython=True)
def iterative_proportional_fitting_AB_AC_BC_no_zeros(cont_cube, delta=0.0001):
    """
    Computes the maximum likelihood estimates of a 2X2X2 table under the hypothesis no three-factor interaction.
    1 and 2 together.
    The formula for the MLEs are obtained via the iterative proportional fitting proposed in Bishop et Al. Discrete multivariate analysis
    chapter 3.
    ----------
    cont_cube (2X2X2 numpy array) : Contingency cube of the three variables
    delta (float) : If the difference between the last estimates and the current estimates, we accept the estimates.
    Returns (2X2X2 numpy array) : Contingency cube with the expected values under the hypothesis of no three-factor interaction
    -------
    """

    xij_ = np.sum(cont_cube, axis=0)
    if np.count_nonzero(xij_) != 4:
        return None
    xi_k = np.sum(cont_cube, axis=2).T
    if np.count_nonzero(xi_k) != 4:
        return None
    x_jk = np.sum(cont_cube, axis=1).T
    if np.count_nonzero(x_jk) != 4:
        return None

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)
        mij_ = np.sum(mijk, axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xij_[i,j] / mij_[i, j]

        mi_k = np.sum(mijk, axis=2).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xi_k[i,k] / mi_k[i, k]

        m_jk = np.sum(mijk, axis=1).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * x_jk[j,k] / m_jk[j, k]
        if np.all(mijk.flatten() > delta):
            if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
                #print('Converged!')
                break
        else :
            return None


    return mijk

def iterative_proportional_fitting_AC_BC(cont_cube, delta=0.01):
    "See the equivalent MLE function above"

    #xij_ = np.sum(cont_cube, axis=0)
    xi_k = np.sum(cont_cube, axis=2).T
    x_jk = np.sum(cont_cube, axis=1).T

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)
        mij_ = np.sum(mijk, axis=0)

        #for i in range(2):
        #    for j in range(2):
        #        for k in range(2):
        #            mijk[k,i,j] = mijk[k,i,j] * xij_[i,j] / mij_[i, j]

        mi_k = np.sum(mijk, axis=2).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xi_k[i,k] / mi_k[i, k]

        m_jk = np.sum(mijk, axis=1).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * x_jk[j,k] / m_jk[j, k]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def iterative_proportional_fitting_AB_BC(cont_cube, delta=0.01):
    "See the equivalent MLE function above"
    xij_ = np.sum(cont_cube, axis=0)
    #xi_k = np.sum(cont_cube, axis=2).T
    x_jk = np.sum(cont_cube, axis=1).T

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)
        mij_ = np.sum(mijk, axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xij_[i,j] / mij_[i, j]

        mi_k = np.sum(mijk, axis=2).T

        #for i in range(2):
        #    for j in range(2):
        #        for k in range(2):
        #            mijk[k,i,j] = mijk[k,i,j] * xi_k[i,k] / mi_k[i, k]

        m_jk = np.sum(mijk, axis=1).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * x_jk[j,k] / m_jk[j, k]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def iterative_proportional_fitting_AB_AC(cont_cube, delta=0.01):
    "See the equivalent MLE function above"
    xij_ = np.sum(cont_cube, axis=0)
    xi_k = np.sum(cont_cube, axis=2).T
    #x_jk = np.sum(cont_cube, axis=1).T

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)
        mij_ = np.sum(mijk, axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xij_[i,j] / mij_[i, j]

        mi_k = np.sum(mijk, axis=2).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xi_k[i,k] / mi_k[i, k]

        m_jk = np.sum(mijk, axis=1).T

        #for i in range(2):
        #    for j in range(2):
        #        for k in range(2):
        #            mijk[k,i,j] = mijk[k,i,j] * x_jk[j,k] / m_jk[j, k]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def iterative_proportional_fitting_BC_A(cont_cube, delta=0.01):
    "See the equivalent MLE function above"
    x_jk = np.sum(cont_cube, axis=1).T
    xi__ = np.sum(np.sum(cont_cube, axis=0), axis=1)

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)

        m_jk = np.sum(mijk, axis=1).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * x_jk[j,k] / m_jk[j, k]

        mi__ = np.sum(np.sum(mijk, axis=0), axis=1)

        for i in range(2):
           for j in range(2):
               for k in range(2):
                   mijk[k,i,j] = mijk[k,i,j] * xi__[i] / mi__[i]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def iterative_proportional_fitting_AB_C(cont_cube, delta=0.01):
    "See the equivalent MLE function above"
    xij_ = np.sum(cont_cube, axis=0)
    x__k = np.sum(np.sum(cont_cube, axis=2), axis=1)

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)

        mij_ = np.sum(mijk, axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xij_[i,j] / mij_[i,j]

        m__k = np.sum(np.sum(mijk, axis=2), axis=1)

        for i in range(2):
           for j in range(2):
               for k in range(2):
                   mijk[k,i,j] = mijk[k,i,j] * x__k[k] / m__k[k]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def iterative_proportional_fitting_AC_B(cont_cube, delta=0.01):
    "See the equivalent MLE function above"
    xi_k = np.sum(cont_cube, axis=2).T
    x_j_ = np.sum(np.sum(cont_cube, axis=1), axis=0)

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)

        mi_k = np.sum(mijk, axis=2).T

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k,i,j] = mijk[k,i,j] * xi_k[i,k] / mi_k[i,k]

        m_j_ = np.sum(np.sum(mijk, axis=1), axis=0)

        for i in range(2):
           for j in range(2):
               for k in range(2):
                   mijk[k,i,j] = mijk[k,i,j] * x_j_[j] / m_j_[j]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def iterative_proportional_fitting_ind(cont_cube, delta=0.01):
    "See the equivalent MLE function above"
    xi__ = np.sum(np.sum(cont_cube, axis=0), axis=1)
    x_j_ = np.sum(np.sum(cont_cube, axis=1), axis=0)
    x__k = np.sum(np.sum(cont_cube, axis=2), axis=1)

    # initialize the MLE at 1 for every cell
    mijk = np.ones((2,2,2))

    while True:
        old_mijk = np.copy(mijk)

        mi__ = np.sum(np.sum(mijk, axis=0), axis=1)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k, i, j] = mijk[k, i, j] * xi__[i] / mi__[i]

        m_j_ = np.sum(np.sum(mijk, axis=1), axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k, i, j] = mijk[k, i, j] * x_j_[j] / m_j_[j]

        m__k = np.sum(np.sum(mijk, axis=2), axis=1)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    mijk[k, i, j] = mijk[k, i, j] * x__k[k] / m__k[k]

        if np.all(np.abs(mijk.flatten() - old_mijk.flatten()) < delta):
            break

    return mijk

def ipf_ABC_ABD_ACD_BCD_no_zeros(hyper_cont_cube, delta=0.0001):
    xijk_ = np.sum(hyper_cont_cube, axis=0)
    if np.count_nonzero(xijk_) != 8:
        return None
    xij_l = np.sum(hyper_cont_cube, axis=1)
    if np.count_nonzero(xij_l) != 8:
        return None
    xi_kl = np.sum(hyper_cont_cube, axis=3)
    if np.count_nonzero(xi_kl) != 8:
        return None
    x_jkl = np.sum(hyper_cont_cube, axis=2)
    if np.count_nonzero(x_jkl) != 8:
        return None

    # initialize the MLE at 1 for every cell
    mijkl = np.ones((2, 2, 2, 2))

    while True:
        old_mijkl = np.copy(mijkl)
        mijk_ = np.sum(mijkl, axis=0)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        mijkl[l, k, i, j] = mijkl[l, k, i, j] * xijk_[k, i, j] / mijk_[k, i, j]

        if not np.all(mijkl.flatten() != 0):
            return None

        mij_l = np.sum(mijkl, axis=1)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        mijkl[l, k, i, j] = mijkl[l, k, i, j] * xij_l[l, i, j] / mij_l[l, i, j]

        if not np.all(mijkl.flatten() != 0):
            return None

        mi_kl = np.sum(mijkl, axis=3)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        mijkl[l, k, i, j] = mijkl[l, k, i, j] * xi_kl[l, k, i] / mi_kl[l, k, i]

        if not np.all(mijkl.flatten() != 0):
            return None

        m_jkl = np.sum(mijkl, axis=2)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        mijkl[l, k, i, j] = mijkl[l, k, i, j] * x_jkl[l, k, j] / m_jkl[l, k, j]

        if np.all(mijkl.flatten() > delta):
            if np.all(np.abs(mijkl.flatten() - old_mijkl.flatten()) < delta):
                break
        else :
            return None

    return mijkl

if __name__ == '__main__':
    print(mle_2x2_ind(np.array([[18,  0],  [20, 0]])), mle_2x2_ind_vector(np.array([[18,  0],  [20, 0]])))
    exit()

    xijk = np.ones((2,2,2))

    xijk[0, 0, 0] = 156
    xijk[0, 1, 0] = 84
    xijk[0, 0, 1] = 84
    xijk[0, 1, 1] = 156

    xijk[1, 0, 0] = 107
    xijk[1, 1, 0] = 133
    xijk[1, 0, 1] = 31
    xijk[1, 1, 1] = 209

    print(iterative_proportional_fitting_ind(xijk, delta=0.000001))
    #chisq_test_2x2x2_AC_B(xijk)

    #print(iterative_proportional_fitting_ABC(xijk, delta=0.000001))