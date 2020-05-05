__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Multi-Layer Perceptron (MLP) Classification [using TensorFlow 1.x] 
"""
import os, sys
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve


# Load data
def load_data():
    '''
    Load iris data from sklearn's dataset
    :return: X, y_encoded
    '''
    from sklearn.datasets import load_iris
    iris = load_iris() # load iris data
    X = pd.DataFrame(iris.data) # iris's features
    y = pd.DataFrame(iris.target)  # iris's labels

    # Encode label column to one hot encoding
    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    y_encoded = encoder.fit_transform(y)
    return X, y_encoded


# Preprocessing:  Preprocessing data, split into training and test part
def preprocessing(X, y):
    '''
    Preprocessing data, split into training and test part
    :param X:
    :param y:
    :return:
    '''
    # Fill the nan values with the column mean (numericalfeatures)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean())

    X = X.dropna() # drop the nan values of categorical features
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_test, y_train, y_test


# TensorFlow: Define MLP model with 3 hidden layers
def multilayer_perceptron(x, weights, biases):
    '''
    Create a deep neural network multilayer perceptron with 3 hidden layers
    :param x: inputs
    :param weights:
    :param biases:
    :return: output_layer: outputs (i.e. y)
    '''
    # Hidden layer with sigmoid activation
    hidden_layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    hidden_layer_1 = tf.nn.sigmoid(hidden_layer_1)

    # Hidden layer with sigmoid activation
    hidden_layer_2 = tf.add(tf.matmul(hidden_layer_1, weights['h2']), biases['b2'])
    hidden_layer_2 = tf.nn.sigmoid(hidden_layer_2)

    # Hidden layer with relu activation
    hidden_layer_3 = tf.add(tf.matmul(hidden_layer_2, weights['h3']), biases['b3'])
    hidden_layer_3 = tf.nn.relu(hidden_layer_3)

    # Output layer with linear activation
    output_layer = tf.matmul(hidden_layer_3, weights['ot']) + biases['ot']
    return output_layer


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Preprocessing data and split into training and test part
    X_train, X_test, y_train, y_test = preprocessing(X, y)

    # TensorFlow: Model initialization
    num_dimensions = X_train.shape[1]          # number of dimensions (feature's dimension)
    learning_rate = 0.03                       # learning rate for GradientDescentOptimizer
    training_epochs = 1000                      # number of epochs in training stage
    hidden_layers = [20, 20, 10]               # number of units per each hidden layer
    num_classes = y_train.shape[1]             # number of classes
    model_path = os.getcwd() + '/saved_model'  # to save trained model in the current directory

    # TensorFlow: Define placeholders
    W = tf.Variable(tf.zeros([num_dimensions, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))
    x = tf.placeholder(tf.float32, [None, num_dimensions])   # None means that it can be any value
    y_ = tf.placeholder(tf.float32, [None, num_classes])     # None means that it can be any value

    # TensorFlow: Neural network mlp weights
    weights = {
        'h1': tf.Variable(tf.truncated_normal([num_dimensions,   hidden_layers[0]])),
        'h2': tf.Variable(tf.truncated_normal([hidden_layers[0], hidden_layers[1]])),
        'h3': tf.Variable(tf.truncated_normal([hidden_layers[1], hidden_layers[2]])),
        'ot': tf.Variable(tf.truncated_normal([hidden_layers[2], num_classes]))
    }

    # TensorFlow: Neural network mlp biases
    biases = {
        'b1': tf.Variable(tf.truncated_normal([hidden_layers[0]])),
        'b2': tf.Variable(tf.truncated_normal([hidden_layers[1]])),
        'b3': tf.Variable(tf.truncated_normal([hidden_layers[2]])),
        'ot': tf.Variable(tf.truncated_normal([num_classes]))
    }

    # TensorFlow: Call the defined model
    y = multilayer_perceptron(x, weights, biases)

    # TensorFlow: Create the cost function and optimizer
    cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
    training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

    # TensorFlow: Model evaluation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
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
    cost_history = np.empty(shape=[1], dtype=float)
    train_accuracy_history = np.empty(shape=[1], dtype=float)
    test_accuracy_history = np.empty(shape=[1], dtype=float)

    for epoch in range(training_epochs):
        sess.run(training_step, feed_dict={x: X_train, y_: y_train})

        cost = sess.run(cost_function, feed_dict={x: X_train, y_: y_train})
        cost_history = np.append(cost_history, cost)

        train_accuracy = sess.run(accuracy, feed_dict={x: X_train, y_: y_train})
        train_accuracy_history = np.append(train_accuracy_history, train_accuracy)

        test_accuracy = sess.run(accuracy, feed_dict={x: X_test, y_: y_test})
        test_accuracy_history = np.append(test_accuracy_history, test_accuracy)

        print('Epoch:{:<5} ->    Training_Accuracy={:.5f}       Training_Cost={:.5f}       Test_Accuracy={:.5f}'
              .format(epoch, round(train_accuracy, 5), round(cost, 5), round(test_accuracy, 5)))

    # TensorFlow: Save the model
    saver.save(sess, model_path)

    # TensorFlow: Restore the saved model
    #saver.restore(sess, model_path')

    # TensorFlow: Evaluation the model
    # Summarize result
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('MLPClassifier:')
    print('Test_Accuracy: ', (sess.run(accuracy, feed_dict={x: X_test, y_: y_test})))

    # Plot settings
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the accuracy for the model
    axes[0].plot(train_accuracy_history, label='{} = {}'.format('TRAIN ACCURACY', round(train_accuracy_history[-1],3)))
    axes[0].plot(test_accuracy_history, label='{} = {}'.format('TEST ACCURACY', round(test_accuracy_history[-1], 3)))
    axes[0].set_xlim([0.0, training_epochs])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('MODEL ACCURACY (TRAIN-TEST)', fontsize=12)
    axes[0].legend(loc="lower right")

    # plot the cost for the model
    axes[1].plot(cost_history, label='{} = {}'.format('COST', round(cost_history[-1], 3)), color='r')
    axes[1].set_xlim([0.0, training_epochs])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Cost')
    axes[1].set_title('COST FUNCTION', fontsize=12)
    axes[1].legend(loc="lower left")

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_mlpclassification.png', bbox_inches='tight')
    plt.show()