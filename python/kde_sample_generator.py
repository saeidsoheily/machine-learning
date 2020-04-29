__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Kernel Density Estimation
"""
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

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


# Generate random samples using Kernel Density Estimation (KDE)
def generate_samples(X, size=100):
    '''
    Generate new sample from the same distribution of original data
    :param X: the original data
    :param size: size of new samples
    :return: data: new sampled data
    '''
    kde = KernelDensity(kernel='gaussian', bandwidth=0.01) # kernel density estimation (0.01: bandwidth of the kernel)
    kde.fit(X)  # fit the kernel density model on the data

    data = kde.sample(size) # generate new random samples from the model
    return kde, data


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Generate a sample wareform time series data
    data = generate_data(size=1000, length=50)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot the generated synthetic data
    plot_data(data, axes[0, 0], title='ORIGINAL DATA')
    density, bins, patches = axes[0, 1].hist(x=data, bins=200, density=True) # plot the histogram (an approximate representation of data distribution)
    axes[0, 1].set_title('HISTOGRAM (ORIGINAL DATA)', fontsize=12)

    # Generate new sample from the same distribution of original data
    kde, new_data = generate_samples(data, size=1000)

    # Plot the new sampled data
    plot_data(new_data, axes[1, 0], title='SAMPLED DATA')
    density, bins, patches = axes[1, 1].hist(x=new_data, bins=200, density=True) # plot the histogram (an approximate representation of data distribution)
    axes[1, 1].set_title('HISTOGRAM (SAMPLED DATA)', fontsize=12)

    # To save the plot locally
    plt.savefig('kde_new_samples.png', bbox_inches='tight')
    plt.show()