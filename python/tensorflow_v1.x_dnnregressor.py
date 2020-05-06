__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: DNN Regression [using TensorFlow 1.x] 
"""
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.data import Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load data
def load_data():
    '''
    Load california housing data from sklearn's datasets
    :return: X, y
    '''
    from sklearn.datasets import fetch_california_housing
    dataset = fetch_california_housing() # load dataset
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)     # features
    y = pd.DataFrame(dataset.target, columns=['median_house_value'])  # target (label)
    return X, y


# Preprocessing:  Preprocessing data, split into training, validation and test part
def preprocessing(X, y):
    '''
    Preprocessing data, split into training, validation and test part
    :param X, y:
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''
    # Fill the nan values with the column mean (numericalfeatures)
    for col in X.columns:
        X[col] = X[col].fillna(X[col].mean())

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_val, X_test, y_train, y_val, y_test


# TensorFlow: Input function
def tf_input_fn(features, label, buffer_size=10000, batch_size=1, shuffle=True, num_epochs=None):
    '''
    Create input function for TensorFlow model
    :param features: pandas dataframe
    :param label: pandas dataframe
    :param buffer_size: shuffle buffer size
    :param batch_size:
    :param shuffle:
    :param num_epochs: (None = repeat indefinitely)
    :return: next batch data
    '''
    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, label))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified
    if shuffle:
        ds = ds.shuffle(buffer_size) # randomly shuffles the elements of dataset

    # Return the next batch of data
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Preprocessing data and split into training, validation and test part
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(X, y)

    # TensorFlow: Initialization
    label = y_train.columns[0] # label column name
    buffer_size = X_train.shape[0] # shuffle buffer size
    learning_rate = 0.01
    batch_size = 100  # number of instances to be read each time (i.e. each iteration)
    iter_number = 500 # number of steps in training stage (a training step: a forward and backward pass using a batch)
    hidden_layers = [10, 5] # number of units per each hidden layer
    clipping_ratio = 5.0 # ratio to clip gradients in optimizer before applying them

    # TensorFlow: Construct the TensorFlow feature columns
    feat_cols = [tf.feature_column.numeric_column(c) for c in X_train.columns]

    # TensorFlow: Create the input function
    training_input_fn = lambda: tf_input_fn(X_train, y_train[label], buffer_size=buffer_size, batch_size=batch_size)
    training_input_fn_ = lambda: tf_input_fn(X_train, y_train[label], shuffle=False, num_epochs=1)
    validation_input_fn = lambda: tf_input_fn(X_val, y_val[label], shuffle=False, num_epochs=1)
    test_input_fn = lambda: tf_input_fn(X_test, y_test[label], shuffle=False, num_epochs=1)

    # TensorFlow: Create the optimizer for DNN classifier
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, clipping_ratio) # clips gradients before applying them

    # TensorFlow: Create the DNN classifier model
    model = tf.estimator.DNNRegressor(feature_columns=feat_cols, hidden_units=hidden_layers, optimizer=optimizer)

    # TensorFlow: Train the model
    periods = 50 # to evaluate training and validation set during the training process
    training_rmse_history = []
    validation_rmse_history = []
    steps_per_period = iter_number / periods
    for period in range(periods):
        model.train(input_fn=training_input_fn, steps=steps_per_period)

        # TensorFlow: Evaluation the model (training and validation set)
        predictions_training = model.predict(input_fn=training_input_fn_)
        predictions_training = np.array([item['predictions'][0] for item in predictions_training])
        training_rmse = math.sqrt(mean_squared_error(predictions_training, y_train)) # training loss
        training_rmse_history.append(training_rmse) # add the loss value to the history

        predictions_validation = model.predict(input_fn=validation_input_fn)
        predictions_validation = np.array([item['predictions'][0] for item in predictions_validation])
        validation_rmse = math.sqrt(mean_squared_error(predictions_validation, y_val)) # validation loss
        validation_rmse_history.append(validation_rmse) # add the loss value to the history

        print('Period:{:<5} ->    Training_RMSE={:.5f}       Validation_RMSE={:.5f}'
              .format(period, round(training_rmse, 5), round(validation_rmse, 5)))

    # TensorFlow: Evaluation the model (test set)
    predictions_test = model.predict(input_fn=test_input_fn)
    predictions_test = np.array([item['predictions'][0] for item in predictions_test])
    test_rmse = math.sqrt(mean_squared_error(predictions_test, y_test))

    # Summarize result
    print('DNNRegressor: RMSE={}'.format(round(test_rmse, 5)))

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the error for the model
    axes[0].plot(training_rmse_history, label='{} = {}'.format('RMSE (TRAINING)', round(training_rmse_history[-1],3)))
    axes[0].plot(validation_rmse_history, label='{} = {}'.format('RMSE (VALIDATION)', round(validation_rmse_history[-1], 3)))
    axes[0].set_xlim([0.0, periods])
    axes[0].set_xlabel('Periods')
    axes[0].set_ylabel('Root Mean Squared Error')
    axes[0].set_title('COST FUNCTION (TRAINING vs VALIDATION)', fontsize=12)
    axes[0].legend(loc="lower left")

    # Plot regressor
    axes[1].set_xlabel('Median House Value (ACTUAL)')
    axes[1].set_ylabel('Median House Value (PREDICTED)')
    axes[1].set_title('DNN REGRESSOR', fontsize=12)
    axes[1].scatter(np.array(y_test[label]), np.array(predictions_test))
    lims = [0, max(np.max(y_test[label]), np.max(predictions_test))]
    axes[1].set_xlim(lims)
    axes[1].set_ylim(lims)
    _ = axes[1].plot(lims, lims, '--r')

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_dnnregression.png', bbox_inches='tight')
    plt.show()