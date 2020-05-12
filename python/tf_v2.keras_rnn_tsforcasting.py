__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms:  TensorFlow Keras RNN-LSTM (Time Series Forcasting) [using TensorFlow 2.x-Keras] 
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Load data
def load_data():
    '''
    Load weather temperature
    :return: ts: weather temperature time series
    '''
    zip_path = tf.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    df = pd.read_csv(os.path.splitext(zip_path)[0])  # read dataframe from google storage
    ts = df['T (degC)']  # time series' data (temperature)
    return ts.values


# Preprocessing data, split into training, validation and test part
def preprocessing_data(ts):
    '''
    Data preprocessing for weather temperature time series
    - To make the prediction, 30 instances of observations are considered to train the model.
    - The target is the weather temperature of the next time stamp.
    :param ts: time series data
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''
    X = []
    y = []
    history = 30
    for i in range(history, len(ts)):
        indices = range(i - history, i)
        X.append(np.reshape(ts[indices], (history, 1))) # historical temperature data
        y.append(ts[i]) # target

    X_train, X_val, y_train, y_val = train_test_split(np.array(X), np.array(y), train_size=0.8)
    X_train, X_test, y_train, y_test = train_test_split(np.array(X), np.array(y), train_size=0.8)
    return X_train, X_val, X_test, y_train, y_val, y_test


# Plot ts time series, its target and its predicted value
def plot_ts(axes, title, X_train, y_train, y_pred):
    '''
    Plot time series
    :param axes: plot axes
    :param title
    :param X_train: time series data
    :param y_train: target
    :param y_pred: predicted value
    :return:
    '''
    axes.plot(X_train, marker='o', label='HISTORY')
    axes.plot(len(X_train) + 1, y_train, marker='+', markersize=12, color='g', linestyle='None', label='TRUE')
    y_pred_avg = np.mean(X_train)  # mean of historical values (i.e. moving average)
    axes.plot(len(X_train) + 1, y_pred_avg, marker='x', markersize=12, color='y', linestyle='None', label='PRED (AVG)')
    axes.plot(len(X_train) + 1, y_pred, marker='*', markersize=10, color='r', linestyle='None', label='PRED (RNN)')
    axes.set_xlim([0, len(X_train) + 5])
    axes.set_xlabel('Day')
    axes.set_ylabel('Temperature')
    axes.set_title('TIME SERIES FORCASTING (SAMPLE {})'.format(title), fontsize=12)
    axes.legend(loc='upper left')
    return


# TensorFlow: Input function
def df_to_dataset(features, label, shuffle=True, batch_size=64):
    '''
    Create tensorflow dataset (input function for the model) from pandas df
    :param features: pandas dataframe
    :param label: pandas dataframe
    :param batch_size:
    :param shuffle:
    :return: batch data
    '''
    # Construct a dataset, and configure batching/repeating.
    ds = tf.data.Dataset.from_tensor_slices((features, label))  # warning: 2GB limit
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))

    ds = ds.batch(batch_size).repeat()
    return ds


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load a sample data
    ts = load_data()

    # Preprocessing data and split into training, validation and test part
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing_data(ts)

    # Recurrent Neural Network (RNN): Long Short Term Memory (LSTM)
    # TensorFlow: Initialization
    batch_size = 64  # number of instances to be read each time (i.e. each iteration)
    epochs = 20  # number of epochs
    steps = 200  # number of steps in training stage per epoch

    # TensorFlow: Create the input function
    train_ds = df_to_dataset(X_train, y_train, batch_size=batch_size)  # train dataset
    val_ds = df_to_dataset(X_val, y_val, shuffle=False)  # validation dataset
    test_ds = df_to_dataset(X_test, y_test, shuffle=False)  # test dataset

    # TensorFlow: Create the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=15,
                             recurrent_activation='sigmoid',
                             return_sequences=True,
                             input_shape=X_train.shape[-2:]),
        tf.keras.layers.LSTM(units=15,
                             recurrent_activation='sigmoid'),
        tf.keras.layers.Dense(1)])

    model.compile(optimizer='adam', loss='mae')  # compile model

    # Train LSTM model
    model.fit(train_ds,
              epochs=epochs,
              steps_per_epoch=steps,
              validation_data=val_ds, validation_steps=50)

    # Plot settings
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # TensorFlow: Evaluation
    # Plot predicting values (2 samples of training data)
    inx = 0  # axes index
    for x, y in train_ds.take(2):
        y_true = y
        y_pred = model.predict(x)
        plot_ts(axes[0, inx], 'TRAINING', x[0], y_true[0], y_pred[0])  # plot sample (training data)
        inx += 1

    # Plot predicting values (2 samples of validation data)
    inx = 0  # axes index
    for x, y in val_ds.take(2):
        y_true = y
        y_pred = model.predict(x)
        plot_ts(axes[1, inx], 'VALIDATION', x[0], y_true[0], y_pred[0])  # plot sample (validation data)
        inx += 1

    # To save the plot locally
    plt.savefig('tensorflow_keras_tsforcasting.png', bbox_inches='tight')
    plt.show()