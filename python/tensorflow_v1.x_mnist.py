__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Deep Learning Image Classification [using TensorFlow 1.x] (MNIST Digits Classification) 
"""
import numpy as np
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import accuracy_score, confusion_matrix


# Load data
def load_data():
    '''
    Load MNIST dataset
    :return: mnist, img_size, num_classes:
    '''
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    img_size = 28  # number of pixels in each dimension of an image
    num_classes = 10 # number of classes, one class for each of 10 digits
    return mnist, img_size, num_classes


# Plot sample data
def plot_sample_data(images, labels, img_size, axes, tag, color_map, onehotencoder_label=False):
    '''
    Plot 15 sample data from mnist dataset using gridspec
    :param images:
    :param labels:
    :param img_size:
    :param axes:
    :param tag: plot title
    :param color_map:
    :param onehotencoder_label:
    :return:
    '''
    # Plot original images
    inner_subplot = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=5, subplot_spec=axes, wspace=0.25, hspace=0.25)
    for row in range(3):
        for col in range(5):
            k = row * 3 + col
            ax = plt.Subplot(fig, inner_subplot[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            if onehotencoder_label:
                ax.set_title(tag+str(np.argmax(labels[k])), fontsize=12)     # title: image label
            else:
                ax.set_title(tag + str(labels[k]), fontsize=12)  # title: image label
            ax.imshow(images[k].reshape(img_size, img_size), aspect='auto', cmap=color_map) # show digit's image
            fig.add_subplot(ax)
    return


# Generate one batch of data for training model
def next_batch(batch_size, data, labels, num_classes):
    '''
    Generate a batch by returnning batch_size of random data samples and labels
    :param batch_size:
    :param data:
    :param labels:
    :param num_classes:
    :return:
    '''
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:batch_size]
    batch_data = data[idx]
    batch_labels = labels[idx]
    batch_labels = np.asarray(batch_labels).reshape(batch_size, num_classes)
    return batch_data, batch_labels


# Plot learned weights
def plot_learned_weights(w, img_size, axes, tag):
    '''
    Plot the learned weights for each class
    :param w: learned weights
    :param img_size: dimension size of an image
    :param axes:
    :param tag: plot title
    :return:
    '''
    inner_subplot = gridspec.GridSpecFromSubplotSpec(nrows=3, ncols=5, subplot_spec=axes, wspace=0.25, hspace=0.25)
    for row in range(2):
        for col in range(5):
            k = row * 5 + col
            ax = plt.Subplot(fig, inner_subplot[row*2, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(tag + str(k), fontsize=12)  # title: image label
            image = w[:, k].reshape(img_size, img_size)
            ax.imshow(image, vmin=np.min(w), vmax=np.max(w), cmap='seismic')
            fig.add_subplot(ax)
    return


# Plot confusion matrix
def plot_confusion_matrix(cls_true, cls_pred, accuracy_ture_pred, axes, num_classes):
    '''
    Plot seaborn heatmap of confusion matrix
    :param cls_true:
    :param cls_pred:
    :param accuracy_ture_pred:
    :param axes:
    :param num_classes:
    :return:
    '''
    cm = confusion_matrix(y_true=cls_true, y_pred=cls_pred)  # confusion matrix using sklearn
    inner_subplot = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=axes)
    ax = plt.Subplot(fig, inner_subplot[0, 0])
    ax.set_xticks(np.arange(num_classes))  # set ticks' marks
    ax.set_yticks(np.arange(num_classes))  # set ticks' marks
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('CONFUSION MATRIX [ACCURACY={}]'.format(accuracy_ture_pred))
    fig.add_subplot(ax)
    sns.set(font_scale=1.2)  # for label size
    color_map = sns.cubehelix_palette(dark=0, light=0.95, as_cmap=True)  # color_map for seaborn plot
    sns.heatmap(cm, cmap=color_map, annot=True, annot_kws={"size": 12}, fmt="d")  # plot confusion matrix heatmap
    return


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load mnist digits dataset
    mnist, img_size, num_classes = load_data()

    # Initial settings
    batch_size = 100    # to use only 200 data to train at each iteration of the optimizer
    iter_number = 5000  # number of iterations of training steps

    # Plot settings
    fig = plt.figure(figsize=(17, 9))  # set figure size
    axes = gridspec.GridSpec(nrows=2, ncols=2, wspace=0.15, hspace=0.15) # set figure shape
    axes.update(top=0.95, left=0.05, right=0.95, bottom=0.05) # set tight_layout()

    # Plot sample actual data
    plot_sample_data(mnist.test.images, mnist.test.labels, img_size, axes[0],
                     tag='ACTUAL: ',
                     color_map='viridis',
                     onehotencoder_label=True)  # plot original image

    # TensorFlow: Define placeholders
    x = tf.placeholder(tf.float32, [None, img_size*img_size])  # flatten image
    y_true = tf.placeholder(tf.float32, [None, num_classes])   # actual one hot encoded label
    y_true_cls = tf.placeholder(tf.int64, [None, 1])           # actual class label (e.g. 6)

    # TensorFlow: Define variabels
    weights = tf.Variable(tf.zeros([img_size*img_size, num_classes]))
    biases = tf.Variable(tf.zeros([num_classes]))

    # TensorFlow: Create (logistic regression) model
    logits = tf.matmul(x, weights) + biases  # y = wx + b

    y_pred = tf.nn.softmax(logits)  # transfer the logits to range [0,1], sum equal to one
    y_pred_cls = tf.argmax(y_pred, axis=1)  # predicted class label (argmax returns index of the highest value)

    # TensorFlow: Define cost function (method 1)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_true)
    cost = tf.reduce_mean(cross_entropy)  # compute the average of all cross_entropies

    # TensorFlow: Define cost function (method 2)
    # cost = -tf.reduce_sum(y_true * tf.log(y_pred))

    # TensorFlow: Define optimization method
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3).minimize(cost)

    # TensorFlow: Model evaluation
    correct_prediction = tf.equal(y_pred_cls, y_true_cls)  # correct_predictions are Boolean
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TensorFlow: Run the defined tensorflow graph
    sess = tf.Session()  # create a session

    # TensorFlow: Initialize the variables
    init = tf.global_variables_initializer()

    sess.run(init) # execute the initializer

    # TensorFlow: Training...
    for _ in range(iter_number): # iterations
        x_batch, y_true_batch = next_batch(batch_size, mnist.train.images, mnist.train.labels, num_classes)
        sess.run(optimizer,
                 feed_dict= {x: x_batch,
                             y_true:y_true_batch})

    # Compute accuracy_score
    cls_true = np.argmax(mnist.test.labels, axis=1).reshape(-1, 1) # true classifications for the test-set

    # Predicted classifications for the test-set
    cls_pred = sess.run(y_pred_cls,
                        feed_dict={x: mnist.test.images,
                                   y_true: mnist.test.labels,
                                   y_true_cls: np.argmax(mnist.test.labels, axis=1).reshape(-1, 1)})
    accuracy_ture_pred = accuracy_score(cls_true, cls_pred)

    # Plot sample predicted data
    plot_sample_data(mnist.test.images, cls_pred, img_size, axes[1],
                     tag='PREDICTED: ',
                     color_map='binary',
                     onehotencoder_label=False)  # plot original image

    # Plot learned weights for the classes
    learned_weights = sess.run(weights)  # weights from the TensorFlow variable
    plot_learned_weights(learned_weights, img_size, axes[2], tag='WEIGHTS: ')

    # Plot confusion matrix
    plot_confusion_matrix(cls_true, cls_pred, accuracy_ture_pred, axes[3], num_classes)

    sess.close()  # close the tensorflow session

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_mnist.png', bbox_inches='tight')
    plt.show()