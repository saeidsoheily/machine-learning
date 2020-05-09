__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Generative Adversarial Network for New Sample Generating [using TensorFlow 1.x]
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


# GAN: Generator network implementation (with 2 hidden layers)
def generator(Z, n_dim, reuse=False):
    '''
    Generator network of GAN
    :param Z: input random samples
    :param n_dim: dimension of the real dataset to be learned
    :param reuse: used for reusing same layers
    :return: output_layer: n-dimensional vector corresponds to dimensions of real dataset which should be learned
    '''
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()

        hidden_layers = [20, 20] # hidden layers of a fully connected neural network
        hidden_layer1 = tf.layers.dense(Z, hidden_layers[0], activation=tf.nn.leaky_relu)
        hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_layers[1], activation=tf.nn.leaky_relu)
        output_layer = tf.layers.dense(hidden_layer2, n_dim)
    return output_layer


# GAN: Discriminator network implementation
def discriminator(X, n_dim, reuse=False):
    '''
    Discriminator network of GAN
    :param X:
    :param n_dim: dimension of the real dataset to be learned
    :param reuse: used for reusing same layers
    :return:
    '''
    with tf.variable_scope("discriminator") as scope: # to recognise in visualise graph in TensorBoard
        if reuse:
            scope.reuse_variables()

        hidden_layers = [20, 20] # hidden layers of a fully connected neural network
        hidden_layer1 = tf.layers.dense(X, hidden_layers[0], activation=tf.nn.leaky_relu)
        hidden_layer2 = tf.layers.dense(hidden_layer1, hidden_layers[1], activation=tf.nn.leaky_relu)
        hidden_layer3 = tf.layers.dense(hidden_layer2, n_dim)
        output_layer = tf.layers.dense(hidden_layer1, 1)
    return output_layer, hidden_layer3


# Return a batch of real data
def sample_X(data, batch_size):
    '''
    Return a sample of real data
    :param data: real data
    :param batch_size:
    :return:
    '''
    if batch_size > len(data):
        repeat_times = np.ceil(batch_size/len(data))
        sample_x = np.array(shuffle(np.repeat(data, repeat_times, axis=0))[:batch_size])
    else:
        sample_x = np.array(shuffle(data)[:batch_size])
    return sample_x


# Generate a random sample data (fake data)
def sample_Z(size, n_dim):
    '''
    Generate a fake sample of data
    :param size: size of samples
    :param n_dim:
    :return:
    '''
    return np.random.uniform(-1., 1., size=[size, n_dim])


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Generate a sample wareform time series data
    data = generate_data(size=100, length=50)

    # TensorFlow: Model initialization
    sample_size = 100
    n_dim = data.shape[1]  # dimension of the real dataset to be learned
    learning_rate = 0.003 # learning rate for the optimizers
    iter_number = 10001

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Plot the generated synthetic data
    plot_data(data, axes[0, 0], title='ORIGINAL DATA')
    density, bins, patches = axes[0, 1].hist(x=data, bins=200, density=True) # plot the histogram
    axes[0, 1].set_title('HISTOGRAM (ORIGINAL DATA)', fontsize=12)

    # TensorFlow: Generative adversarial network (GAN)
    # TensorFlow: GAN: Defining placeholders
    X = tf.placeholder(tf.float32, [None, n_dim])  # real samples
    Z = tf.placeholder(tf.float32, [None, n_dim])  # random noise samples

    # TensorFlow: GAN: Generator network
    generator_samples = generator(Z, n_dim) # create the graph for generating samples from Generator network

    # TensorFlow: GAN: Discriminator network
    real_logits, r_rep = discriminator(X, n_dim)  # feeding real samples to the discriminator network
    faked_logits, g_rep = discriminator(generator_samples, n_dim, reuse=True) # feeding generated samples

    # TensorFlow: GAN: Loss function using Ex[log(D(x))]  + Ez[log(1-D(G(z)))]
    generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=faked_logits,
                                                                            labels=tf.ones_like(faked_logits)))

    discriminator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,
                                                                                labels=tf.ones_like(real_logits)) +
                                        tf.nn.sigmoid_cross_entropy_with_logits(logits=faked_logits,
                                                                                labels=tf.zeros_like(faked_logits)))

    # TensorFlow: GAN: Optimizer
    generator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="generator")
    generator_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate)\
        .minimize(generator_loss, var_list=generator_vars)  # generator training step

    discriminator_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="discriminator")
    discriminator_step = tf.train.RMSPropOptimizer(learning_rate=learning_rate)\
        .minimize(discriminator_loss, var_list=discriminator_vars)  # discriminator training step

    # TensorFlow: Run the defined tensorflow graph
    sess = tf.Session()  # create a session

    # TensorFlow: Initialize the variables
    init = tf.global_variables_initializer()

    # TensorFlow: Create and instance of train.Saver
    saver = tf.train.Saver()

    sess.run(init) # execute the initializer

    # TensorFlow: GAN: Train both generator and discriminator networks in an alternating way
    nd_steps = ng_steps = 2
    for i in range(iter_number):
        X_batch = sample_X(data, sample_size)    # generate a batch of real data
        Z_batch = sample_Z(sample_size, n_dim)   # generate a random sample data (fake data)

        for _ in range(nd_steps):
            _, dloss = sess.run([discriminator_step, discriminator_loss], feed_dict={X: X_batch, Z: Z_batch})

        rrep_dstep, grep_dstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(ng_steps):
            _, gloss = sess.run([generator_step, generator_loss], feed_dict={Z: Z_batch})

        rrep_gstep, grep_gstep = sess.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        # Summarize result
        if i % 100 == 0:
            print('Iteration{: < 6} ->    Discriminator_Loss = {:.3f}       Generator_Loss = {:.3f}'.
                  format(i, round(dloss,3), round(gloss,3)))

    # TensorFlow: GAN: Generate learned samples
    generated_samples = sess.run(generator_samples, feed_dict={Z: Z_batch})

    # TensorFlow: Save the model
    saver.save(sess, './gan_sampling.model')

    # TensorFlow: Restore the saved model (uncomment to restore the saved model)
    #saver.restore(sess, './gan_sampling.model')

    # Plot the new sampled data
    plot_data(generated_samples, axes[1, 0], title='SAMPLED DATA USING GAN')
    density, bins, patches = axes[1, 1].hist(x=generated_samples, bins=200, density=True) # plot the histogram
    axes[1, 1].set_title('HISTOGRAM (SAMPLED DATA)', fontsize=12)

    sess.close()  # close the tensorflow session

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_gan_new_samples.png', bbox_inches='tight')
    plt.show()