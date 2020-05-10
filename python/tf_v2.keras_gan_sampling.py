__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Generative Adversarial Network for New Sample Generating [using TensorFlow 2.x-Keras] 
"""
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle


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
    for i in range(0, size):
        data.append(np.sin(6 * np.pi * x / length) + np.cos(2 * np.pi * x / length) + random.random())
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


# GAN: Generator network implementation (standalone generator model)
def define_generator(n_dim):
    '''
    Define the standalone generator network of GAN
    :param n_dim: number of dimension of data
    :return: generative model
    '''
    generative_model = tf.keras.Sequential()
    generative_model.add(tf.keras.layers.Dense(20,
                                               activation='relu',
                                               kernel_initializer='he_uniform',
                                               input_dim=n_dim))
    generative_model.add(tf.keras.layers.Dense(20,
                                               activation='relu',
                                               kernel_initializer='he_uniform'))
    generative_model.add(tf.keras.layers.Dense(n_dim,
                                               activation='linear'))
    return generative_model


# GAN: Discriminator network implementation (standalone discriminator model)
def define_discriminator(n_dim):
    '''
    Define the standalone discriminator network of GAN
    :param n_dim: number of dimension of data
    :return: discriminator model
    '''
    discriminator_model = tf.keras.Sequential()
    discriminator_model.add(tf.keras.layers.Dense(20,
                                                  activation='relu',
                                                  kernel_initializer='he_uniform',
                                                  input_dim=n_dim))
    discriminator_model.add(tf.keras.layers.Dense(20,
                                                  activation='relu',
                                                  kernel_initializer='he_uniform'))
    discriminator_model.add(tf.keras.layers.Dense(1,
                                                  activation='sigmoid'))

    # Compile discriminator model
    discriminator_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return discriminator_model


# GAN: Generative Adversarial Network (combined generator and discriminator model)
def define_gan(generator, discriminator):
    '''
    Define the combined generator and discriminator model for updating generator model
    :param generator: generator model
    :param discriminator: discriminator model
    :return: gan model
    '''
    # Define GAN model
    gan_model = tf.keras.Sequential()
    gan_model.add(generator)         # add generator model
    discriminator.trainable = False  # make weights in the discriminator not trainable
    gan_model.add(discriminator)     # add discriminator model

    # Compile model
    gan_model.compile(loss='binary_crossentropy', optimizer='adam')
    return gan_model


# Return a batch of real data with class labels 1
def generate_real_samples(data, batch_size):
    '''
    Return a sample of real data
    :param data: real data
    :param batch_size:
    :return:
    '''
    if batch_size > len(data):
        repeat_times = np.ceil(batch_size / len(data))
        sample_x = np.array(shuffle(np.repeat(data, repeat_times, axis=0))[:batch_size])
    else:
        sample_x = np.array(shuffle(data)[:batch_size])

    # Generate class labels 1
    y = np.ones((batch_size, 1))
    return sample_x, y


# Generate a random sample data (fake data)
def generate_fake_samples(n_dim, size, fake_label=True):
    '''
    Generate a fake sample of data
    :param size: size of samples
    :param n_dim:
    :param fake_label
    :return:
    '''
    sample_z = np.random.uniform(-3., 3., size=[size, n_dim])

    # Create class labels
    if fake_label:
        y = np.zeros((size, 1))
    else:
        y = np.ones((size, 1))
    return sample_z, y


# GAN: Evaluate discriminator model and plot data
def summarize_performance(data, epoch, generator, discriminator, n_dim, axes, bt_size=100):
    '''
    Evaluate the accuracy of discriminator model and plot the generated samples
    :param data: original data
    :param epoch:
    :param generator: generator model
    :param discriminator: discriminator model
    :param n_dim: number of dimension of data
    :param axes:
    :param bt_size: size of data to evaluate
    :return:
    '''
    # Generate real samples
    x_real, y_real = generate_real_samples(data, bt_size)

    # Evaluate discriminator on real samples
    _, acc_real = discriminator.evaluate(x_real, y_real, verbose=0)

    # Gnerate fake examples
    x_fake, y_fake = generate_fake_samples(n_dim, bt_size)
    x_fake = generator.predict(x_fake)

    # Evaluate discriminator on fake samples
    _, acc_fake = discriminator.evaluate(x_fake, y_fake, verbose=0)

    # Summarize discriminator performance
    print('Iteration{: < 6} ->    Discriminator_Accuracy (real) = {:.3f}       Discriminator_Accuracy (fake) = {:.3f}'.
        format(epoch, round(acc_real, 3), round(acc_fake, 3)))

    # Plot generated data
    plot_data(x_fake, axes, 'SAMPLED DATA USING GAN (EPOCH:{})'.format(epoch))
    return


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Generate a sample wareform time series data
    data = generate_data(size=100, length=50)

    # TensorFlow: Model initialization
    sample_size = 100
    n_dim = data.shape[1]  # dimension of the real dataset to be learned
    learning_rate = 0.01  # learning rate for the optimizers
    iter_number = 10001
    c = 0  # plot axes index

    # Plot settings
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(20, 6))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib
    axes[0, 1].set_visible(False)
    axes[0, 2].set_visible(False)
    axes[0, 3].set_visible(False)

    # Plot original data
    plot_data(data, axes[0, 0], 'ORIGINAL DATA')

    # TensorFlow: Generative adversarial network (GAN)
    # TensorFlow: GAN: Generator model
    generator = define_generator(n_dim)

    # TensorFlow: GAN: Discriminator model
    discriminator = define_discriminator(n_dim)

    # TensorFlow: GAN: Combined generator and discriminator model
    gan = define_gan(generator, discriminator)

    # TensorFlow: GAN: Train both generator and discriminator networks in an alternating way
    for i in range(iter_number):
        x_real, y_real = generate_real_samples(data, int(sample_size / 2)) # real samples to train discriminator
        x_fake, y_fake = generate_fake_samples(n_dim, int(sample_size / 2)) # fake samples to train discriminator

        # Train discriminator
        discriminator.train_on_batch(x_real, y_real)
        discriminator.train_on_batch(generator.predict(x_fake), y_fake)

        # Generate fake samples as input for generator
        x_gan, y_gan = generate_fake_samples(n_dim, sample_size, fake_label=False)

        # Train generator via the discriminator's error
        gan.train_on_batch(x_gan, y_gan)

        # Summarize result
        if i == 1 or i == int(iter_number / 10) or i == int(iter_number / 2) or i + 1 == iter_number:
            summarize_performance(data, i, generator, discriminator, n_dim, axes[1, c])
            c += 1 # axes index

    # To save the plot locally
    plt.savefig('tensorflow_keras_gan_new_samples.png', bbox_inches='tight')
    plt.show()