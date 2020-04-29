__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: BETA Distribution
"""
import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt

def plot_beta(x, a, b, mu=0, sigma=1, cdf=False, **kwargs):
    '''
    Plots the f distribution function for a given x range, a and b
    If mu and sigma are not provided, standard beta is plotted
    If cdf=True cumulative distribution is plotted
    Passes any keyword arguments to matplotlib plot function
    '''
    if cdf:
        y = ss.beta.cdf(x, a, b, mu, sigma)
    else:
        y = ss.beta.pdf(x, a, b, mu, sigma)
    plt.plot(x, y, **kwargs)


#------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Variables Initialization for Plotting
    x = np.linspace(0, 1, 5000)
    mu = [[5, 4, 2, 3, 5], [1, 2, 3, 4, 5], [1, 1, 1, 2, 3]]
    sigma = [[1, 1, 1, 2, 3], [1, 2, 3, 4, 5], [5, 4, 2, 3, 5]]
    color = ['r', 'b', 'g', 'k', 'm']
    sign = ['>', '=', '<']

    plt.figure(figsize=(14, 9))
    # Plot Probability Density Function (PDF)
    for i in range(3):
        plt.subplot(2, 3, i+1)
        for m,s,c in zip(mu[i], sigma[i], color):
            plot_beta(x, m, s, 0, 1, color=c, lw=2, ls='-', alpha=0.5, label='α={}, β={}'.format(m, s))

        plt.xlabel('θ')
        plt.ylabel('p(θ)')
        plt.title('PDF [Beta distribution (α{}β)]'.format(sign[i]))
        plt.legend()

    # Plot Cumulative Density Function (CDF)
    cdf = True
    for i in range(3):
        plt.subplot(2, 3, (cdf*3)+(i+1))
        for m,s,c in zip(mu[i], sigma[i], color):
            plot_beta(x, m, s, 0, 1, cdf=True, color=c, lw=2, ls='-', alpha=0.5, label='α={}, β={}'.format(m, s))

        plt.xlabel('θ')
        plt.ylabel('P(θ)')
        plt.title('CDF [Beta distribution (α{}β)]'.format(sign[i]))
        plt.legend()

    # To save the plot locally
    plt.savefig('beta_distribution.png', bbox_inches = 'tight')
    plt.show()