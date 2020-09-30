import numpy as np
import scipy as sp
from scipy.stats import chi2
import matplotlib.pyplot as plt
from matplotlib import rc



if __name__ == '__main__':



    x = np.arange(0.0001, 20, 0.001)

    print(1 - chi2.cdf(120, 1))
    plt.plot(x, 1 - chi2.cdf(x, 1), label=r'$\chi^2(\nu = 1)')
    plt.plot(x, chi2.cdf(x, 3))
    plt.plot(x, chi2.cdf(x, 7))
    plt.plot([0, 100], [0.01, 0.01])
    plt.xlim(0, 10)
    plt.ylim(0, 1)
    plt.legend(loc=0)
    plt.show()