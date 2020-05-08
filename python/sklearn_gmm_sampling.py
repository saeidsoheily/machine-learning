__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Gaussian Mixture Model for New Sample Generating [using Sklearn]
"""
import random
import itertools
import numpy as np
from sklearn import mixture
import matplotlib.pyplot as plt


# Generate synthetic data
def generate_data(size, length):
    '''
    Generate waveform times series
    :param size:
    :param length:
    :return:
    '''
    x = np.arange(0, length)
    data = []
    for i in range(0, int(size / 2)):
        data.append(np.sin(4 * np.pi * x / length) + np.cos(2 * np.pi * x / length) + random.random())
    for i in range(int(size / 2), size):
        data.append(-(np.sin(4 * np.pi * x / length) + np.cos(2 * np.pi * x / length) + random.random()))
    return np.array(data)


# Plot data
def plot_data(data, axes, title):
    '''
    Plot the synthetic data
    :param data:
    :param axes: matlibplot axes of plotting data
    :param title: plot title
    :return:
    '''
    axes.plot(data.T)
    axes.set_title(title, fontsize=12)
    return


# Representation of a Gaussian mixture model probability distribution
def gmm(X, axes_a, axes_b):
    '''
    Estimate the best parameters of a Gaussian mixture distribution
    :param data:
    :param axes_a: matlibplot axes of plotting aic
    :param axes_b: matlibplot axes of plotting bic
    :return: best_gmm
    '''

    # Variables Initialization for estimating
    aic = []  # the Akaike information criterion
    bic = []  # the Bayesian information criterion
    lowest_aic = np.infty  # the lowest Akaike information criterion for the model on the input X
    lowest_bic = np.infty  # the lowest Bayesian information criterion for the model on the input X
    n_components_range = range(1, 7)  # the number of mixture components
    cv_types = ['spherical', 'tied', 'diag', 'full']  # the type of covariance parameters to use
    color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue', 'darkorange'])  # colors in plotting bars

    for cv_type in cv_types:
        for n_components in n_components_range:
            gmm = mixture.GaussianMixture(n_components=n_components, covariance_type=cv_type)
            gmm.fit(X)  # fit data to estimate the model parameters using the EM algorithm
            aic.append(gmm.aic(X))
            bic.append(gmm.bic(X))
            if aic[-1] < lowest_aic:
                lowest_aic = aic[-1]
                best_gmm_aic = gmm

            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm_bic = gmm

    aic = np.array(aic)
    bic = np.array(bic)

    # Plot the AIC scores
    bars = []
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(axes_a.bar(xpos, aic[i * len(n_components_range): (i + 1) * len(n_components_range)], width=.2,
                               color=color))
    axes_a.set_xticks(n_components_range)
    axes_a.set_ylim([aic.min() * 1.01 - .01 * aic.max(), aic.max()])
    axes_a.set_title('AIC SCORE PER MODEL', fontsize=12)
    xpos = np.mod(aic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(aic.argmin() / len(n_components_range))
    axes_a.text(xpos, aic.min() * 0.97 + .03 * aic.max(), '*', fontsize=14)
    axes_a.set_xlabel('Number of components')
    axes_a.legend([b[0] for b in bars], cv_types)

    # Plot the BIC scores
    bars = []
    for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
        xpos = np.array(n_components_range) + .2 * (i - 2)
        bars.append(axes_b.bar(xpos, bic[i * len(n_components_range): (i + 1) * len(n_components_range)], width=.2,
                               color=color))
    axes_b.set_xticks(n_components_range)
    axes_b.set_ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
    axes_b.set_title('BIC SCORE PER MODEL', fontsize=12)
    xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 + .2 * np.floor(bic.argmin() / len(n_components_range))
    axes_b.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
    axes_b.set_xlabel('Number of components')
    axes_b.legend([b[0] for b in bars], cv_types)

    if best_gmm_aic != best_gmm_bic:
        print('Warning: The best model is selected based on the Bayesian information criterion!')
    print('Estimate model parameters: \n', best_gmm_aic)
    return best_gmm_bic


# Generate random samples from the fitted Gaussian distribution
def generate_samples(gmm, size=100):
    '''
    Generate new sample from the same distribution of original data
    :param gmm: the estimated gmm
    :param size: size of new samples
    :return: data: new sampled data
    '''
    data = gmm.sample(n_samples=size)
    return data[0]


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Generate a sample wareform time series data
    data = generate_data(size=100, length=50)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 10))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot the generated synthetic data
    plot_data(data, axes[0], title='ORIGINAL DATA')

    # Estimate the best parameters of a Gaussian mixture distribution
    best_gmm = gmm(data, axes[1], axes[2])

    # Generate new sample from the same distribution of original data
    new_data = generate_samples(best_gmm, size=100)

    # Plot the new sampled data
    plot_data(new_data, axes[3], title='SAMPLED DATA')

    # To save the plot locally
    plt.savefig('gmm_new_samples.png', bbox_inches='tight')
    plt.show()
