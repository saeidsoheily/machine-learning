__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: TensorFlow Keras RNN (NLP-Next Word Prediction)  [using TensorFlow 1.x] 
"""
import os
import random
import numpy as np
import collections
import urllib.request
import tensorflow as tf
from termcolor import colored
import matplotlib.pyplot as plt


# Load data
def load_data():
    '''
    read a text story from url
    :return: array of ordered words
    '''
    data = urllib.request.urlopen('http://textfiles.com/stories/alad10.txt') # aladdin and the wonderful lamp story

    content = ''
    for line in data:
        line = line.decode('utf-8', 'replace').replace('\n', ' ').replace('\r', '')
        content += line

    words_list = content.split()
    return np.array(words_list).reshape([-1, ])


# Create a dictionary of words from the words' array
def create_dictionary(words):
    '''
    create a dictionary of words
    :param words:
    :return:
    '''
    # Create the tuples of word frequency counter
    count = collections.Counter(words).most_common()

    # Create a dictionary of (words, order_number)
    dict_word_ordnumber = {}
    for word, _ in count:
        dict_word_ordnumber[word] = len(dict_word_ordnumber)

    # Invert the dictionary
    dict_ordnumber_word = dict(zip(dict_word_ordnumber.values(), dict_word_ordnumber.keys())) # reversed dict
    return dict_word_ordnumber, dict_ordnumber_word


# TensorFlow: Define RNN model with 3 hidden layers
def recurrent_neural_network(x, weights, biases, n_inputs, n_units):
    '''
    Create a recurrent neural network with 3 hidden layers
    :param x: inputs
    :param weights:
    :param biases:
    :param n_inputs: number of rnn inputs
    :param n_units: number of units in rnn cells
    :return: output_layer: outputs (i.e. y)
    '''
    x = tf.reshape(x, [-1, n_inputs]) # reshape input
    x = tf.split(x, n_inputs, 1) # # n_inputs-element of sequence (e.g. [i] [am] [a] >>> [34] [2] [14])

    # Define a 2-layers LSTM with n_hidden units
    rnn_cell = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(n_units),
                                            tf.contrib.rnn.BasicLSTMCell(n_units)])

    output, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)

    # Output layer (n_input outputs exist in rnn_cell, only the last one should be passed as output)
    output_layer = tf.add(tf.matmul(output[-1], weights['ot']) , biases['ot']) # return the last output
    return output_layer


# Create mini-batch labeled data
def create_minibatch(words, words_dict, n_inputs, v_size, batch_size):
    '''
    Create minibatch labeled data
    :param words: array of words
    :param words_dict: dictionary of (words, order_number)
    :param n_inputs: number of inputs
    :param v_size: vocabulary size
    :return:
    '''
    input_words_dvalues_lst = []
    next_word_encoded_lst = []
    input_words_lst = []
    next_word_lst = []
    for i in range(batch_size):
        index = random.randint(0, len(words)-n_inputs-1) # random word index in training data
        input_words_dvalues = [words_dict[str(words[i])] for i in range(index, index + n_inputs)]
        input_words_dvalues = np.reshape(np.array(input_words_dvalues), [n_inputs, 1]) # inputs
        input_words_dvalues_lst.append(input_words_dvalues)

        next_word_encoded = np.zeros([v_size], dtype=int)
        next_word_encoded[words_dict[str(words[index + n_inputs])]] = 1
        next_word_encoded_lst.append(next_word_encoded)

        # To display (print) model evaluation samples
        input_words = [str(words[i]) for i in range(index, index + n_inputs)]  # inputs (string)
        input_words_lst.append(input_words)

        next_word = words[index + n_inputs]  # target (string)
        next_word_lst.append(next_word)
    return input_words_dvalues_lst, next_word_encoded_lst, input_words_lst, next_word_lst


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load a text file
    words = load_data()

    # Create dictionary of words
    words_dict, reversed_words_dict = create_dictionary(words)

    # TensorFlow: Model initialization
    learning_rate = 0.003 # TensorFlow optimizer learning rate
    inputs = 4 # number of inputs
    hidden_layer_units = 128  # number of units in RNN cell
    training_epochs = 50000 # number of epochs (iterations)
    vocabulary_size = len(words_dict) # number of distinc words in data
    mbatch_size = 32
    model_path = os.getcwd() + '/saved_model' # path to save the model

    # TensorFlow: Define placeholders
    x = tf.placeholder(tf.float32, [None, inputs, 1])
    y = tf.placeholder(tf.float32, [None, vocabulary_size])

    # TensorFlow: Define recurrent neural network weights and biases
    weights = {
        'ot': tf.Variable(tf.random_normal([hidden_layer_units, vocabulary_size]))
    }

    biases = {
        'ot': tf.Variable(tf.random_normal([vocabulary_size]))
    }

    # TensorFlow: Define rnn model
    y_ = recurrent_neural_network(x, weights, biases, inputs, hidden_layer_units)

    # TensorFlow: Define loss function and optimiser
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_, labels=y))
    optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost)

    # TensorFlow: Model evaluation
    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # TensorFlow: Create a session to run the defined tensorflow graph
    sess = tf.Session()  # create a session

    # TensorFlow: Initialize the variables
    init = tf.global_variables_initializer()

    # TensorFlow: Create and instance of train.Saver
    saver = tf.train.Saver()

    # TensorFlow: Create an object class for writting summaries
    file_writer = tf.summary.FileWriter(model_path, sess.graph)

    sess.run(init) # execute the initializer

    # TensorFlow: Training...
    cost_history = np.empty(shape=[0], dtype=float) # for display print
    accuracy_history = np.empty(shape=[0], dtype=float) # for display print
    accumulative_cost_history = np.empty(shape=[0], dtype=float) # for cost plot
    accumulative_accuracy_history = np.empty(shape=[0], dtype=float) # for accuracy plot
    for epoch in range(training_epochs):
        X_batch, y_batch, X_words, y_word = create_minibatch(words, words_dict, inputs, vocabulary_size, mbatch_size)

        sess.run(optimizer, feed_dict={x: X_batch, y: y_batch})

        cost_epoch = sess.run(cost, feed_dict={x: X_batch, y: y_batch})
        cost_history = np.append(cost_history, cost_epoch)
        accumulative_cost_history = np.append(accumulative_cost_history, np.mean(cost_history))

        accuracy_epoch = sess.run(accuracy, feed_dict={x: X_batch, y: y_batch})
        accuracy_history = np.append(accuracy_history, accuracy_epoch)
        accumulative_accuracy_history = np.append(accumulative_accuracy_history, np.mean(accuracy_history))

        y_pred = sess.run(y_, feed_dict={x: X_batch, y: y_batch})
        # Summerize result
        if epoch%500 == 0:
            print(colored('Epoch:{:<5}  Accuracy={:.3f}   Cost={:.3f}'.format(epoch,
                                                                              round(np.mean(accuracy_history), 3),
                                                                              round(np.mean(cost_history), 3)), 'red'))
            for mb in range(min(3,mbatch_size)):
                print("...%s [%s] predicted as [%s]" % (
                    ' '.join(X_words[mb]),
                    colored(y_word[mb], 'green'),
                    colored(reversed_words_dict[np.argmax(y_pred[mb])], 'blue'))) # print 3 samples (true vs predicted)
            print()

    # TensorFlow: Save the model
    saver.save(sess, model_path)

    # TensorFlow: Restore the saved model (uncomment the line below to restore the saved model)
    # saver.restore(sess, model_path')

    # TensorFlow: Close the session
    sess.close()

    # Plot settings
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the accuracy for the model
    axes[0].plot(accumulative_accuracy_history,
                 label='{} = {}'.format('ACCURACY', round(accumulative_accuracy_history[-1],3)))
    axes[0].set_xlim([0.0, training_epochs])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('MODEL ACCURACY (TRAINING)', fontsize=12)
    axes[0].legend(loc="lower right")

    # plot the cost for the model
    axes[1].plot(accumulative_cost_history,
                 label='{} = {}'.format('COST', round(accumulative_cost_history[-1], 3)), color='r')
    axes[1].set_xlim([0.0, training_epochs])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cost')
    axes[1].set_title('COST FUNCTION (TRAINING)', fontsize=12)
    axes[1].legend(loc="lower left")

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_rnn_nlp.png', bbox_inches='tight')
    plt.show()