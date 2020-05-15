__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Scikit-Learn Bernoulli Restricted Boltzmann Machine (RBM) Feature Extraction
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neural_network import BernoulliRBM
from sklearn.model_selection import train_test_split


# Load data
def load_data():
    '''
    Load MNIST dataset
    :return: mnist, img_size, num_classes:
    '''
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Reshaping
    img_size = X_train.shape[-1]
    X_train = X_train.reshape((-1, img_size * img_size))
    X_test = X_test.reshape((-1, img_size * img_size))

    # Normalization
    epsilon = 1e-6 # solve divison by zero
    X_train = (X_train - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) + epsilon)  # normalization
    X_test = (X_test - np.min(X_train, axis=0)) / (np.max(X_train, axis=0) + epsilon)  # normalization

    return X_train, X_test, y_train, y_test, img_size


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load mnist data
    X_train, X_test, y_train, y_test, img_size = load_data()

    # Bernoulli Restricted Boltzmann Machine: Initialization
    rbm_n_components = 64 # number of binary hidden units
    rbm_lr = 0.01
    rbm_batch_size = 100
    iter_number = 2

    # Bernoulli Restricted Boltzmann Machine (RBM)
    rbm = BernoulliRBM(n_components=rbm_n_components, # number of binary hidden units
                       learning_rate=rbm_lr, #  learning rate for weight updates
                       batch_size=rbm_batch_size, # number of samples per minibatch
                       n_iter=iter_number, # number of iterations over the training data
                       verbose=True)

    # Fit the model to the data
    rbm_model = rbm.fit(X_train, y_train)

    # Compute the hidden layer activation probabilities: P(h=1|v=X)
    h = rbm_model.transform(X_train) # h: latent representations of the data

    # Summerize result
    print("Shape of data:", X_train.shape)  # (n_samples, n_components)
    print("Shape of data's latent representation:", h.shape) # (n_samples, n_components)

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot the extracted commponents
    plt_dim = int(np.sqrt(rbm_n_components))
    for i, component in enumerate(rbm.components_):
        plt.subplot(plt_dim, plt_dim, i + 1)
        plt.imshow(component.reshape((img_size, img_size)), interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('EXTRACTED COMPONENTS BY RBM', fontsize=12)
    plt.show()