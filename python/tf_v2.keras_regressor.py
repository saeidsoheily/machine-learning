__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms:  TensorFlow Keras Regression [using TensorFlow 2.x-Keras] 
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Load data
def load_data():
    '''
    Load auto mepg data from uci machine learning repository
        1. mpg:           continuous (target)
        2. cylinders:     multi-valued discrete
        3. displacement:  continuous
        4. horsepower:    continuous
        5. weight:        continuous
        6. acceleration:  continuous
        7. model year:    multi-valued discrete
        8. origin:        multi-valued discrete
        9. car name:      string (unique for each instance)
    :return: X, y
    '''
    column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin',
                    'car_name'] # 'car_name': useless
    features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
    label = ['mpg']  # target column

    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data',
                     header=None,
                     sep='\s+',
                     names=column_names)  # load dataset

    df = df.replace('?', np.nan)  # replace values given in to_replace with value
    df = df.dropna()  # drop the nan values of categorical features

    X = df[features]  # features
    y = df[label]  # target (label)
    return X, y


# Preprocessing:  Preprocessing data, split into training, validation and test part
def preprocessing(X, y):
    '''
    Preprocessing data, split into training, validation and test part
    :param X, y:
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''
    # Fill the nan values with the column mean (numericalfeatures)
    float_columns = ['displacement', 'horsepower', 'weight', 'acceleration']
    X[float_columns] = X[float_columns].astype('float32')
    for col in float_columns:
        X[col] = X[col].fillna(X[col].mean())

    # Convert dtypes categorical features to string
    X[['cylinders', 'origin']] = X[['cylinders', 'origin']].astype('str')

    # Normalize the features
    numerical_features = ['displacement', 'horsepower', 'weight', 'acceleration']
    X[numerical_features] = (X[numerical_features] - X[numerical_features].mean()) / X[numerical_features].std()

    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_val, X_test, y_train, y_val, y_test


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
    ds = tf.data.Dataset.from_tensor_slices((dict(features), label))  # warning: 2GB limit
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))

    ds = ds.batch(batch_size)
    return ds


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load data
    X, y = load_data()

    # Preprocessing data and split into training, validation and test part
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(X, y)

    # TensorFlow: Initialization
    learning_rate = 0.001
    batch_size = 100  # number of instances to be read each time (i.e. each iteration)
    iter_number = 200  # number of steps in training stage (a training step: a forward and backward pass using a batch)

    # TensorFlow: Create the TF feature columns for numerical features
    displacement = tf.feature_column.numeric_column('displacement')
    horsepower = tf.feature_column.numeric_column('horsepower')
    weight = tf.feature_column.numeric_column('weight')
    acceleration = tf.feature_column.numeric_column('acceleration')

    # TensorFlow: Create the TF feature columns for categorical features
    cylinders = tf.feature_column.categorical_column_with_hash_bucket('cylinders', hash_bucket_size=10)
    model_year = tf.feature_column.numeric_column('model_year')
    bucketized_model_year = tf.feature_column.bucketized_column(model_year, boundaries=[70, 73, 76, 79, 83])
    origin = tf.feature_column.categorical_column_with_vocabulary_list('origin', ['1', '2', '3'])

    # TensorFlow: Create TF crossed feature (combining features into a single feature)
    crossed_feature_mo = tf.feature_column.crossed_column([bucketized_model_year, origin], hash_bucket_size=100)

    # TensorFlow: Create feature columns by considering both categorical and numerical features [Method 1]
    feat_cols = [displacement, horsepower, weight, acceleration,
                 tf.feature_column.indicator_column(cylinders), # or ...column.embedding_column(cylinders, dimension=5),
                 tf.feature_column.embedding_column(bucketized_model_year, dimension=5),
                 tf.feature_column.indicator_column(origin),
                 tf.feature_column.indicator_column(crossed_feature_mo) # combination of 'model_year' and 'origin'
                 ]

    # TensorFlow: Create the input function
    train_ds = df_to_dataset(X_train, y_train, batch_size=batch_size)  # train dataset
    val_ds = df_to_dataset(X_val, y_val, shuffle=False)  # validation dataset
    test_ds = df_to_dataset(X_test, y_test, shuffle=False)  # test dataset

    # TensorFlow: Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.DenseFeatures(feat_cols),
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.5), # regularization, useful for neural networks (not usefull in this case)
        tf.keras.layers.Dense(128, activation='relu'),
        # tf.keras.layers.Dropout(0.5), # regularization, useful for neural networks (not usefull in this case)
        tf.keras.layers.Dense(1)])

    # TensorFlow: Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss="mean_squared_error",
                  metrics=['mse', 'mae'])  # or metrics=[tf.keras.metrics.MeanSquaredError(), ...]

    # TensorFlow: Train the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', # quantity to be monitored (validation loss)
                                                      verbose=1,  # verbosity mode
                                                      patience=50,  # epochs with no improvement to be stopped
                                                      mode='min',  # ("auto", "min", "max")
                                                      restore_best_weights=True) # restore model weights with best value

    history = model.fit(train_ds,  # (train_features, train_label)
                        validation_data=val_ds,  # (val_features, val_label)
                        callbacks=[early_stopping],  # stop training when a monitored quantity has stopped improving
                        epochs=iter_number)

    # TensorFlow: Save the model
    model.save('tf_keras_regression.model')

    # TensorFlow: Restore the model (uncomment to restore the saved model)
    #model = tf.keras.models.load_model('tf_keras_regression.model')

    # TensorFlow: Evaluation the model (test set)
    loss, mse, mae = model.evaluate(test_ds)  # (test_features, test_label)
    test_predictions = model.predict(test_ds)

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the MSE for the model
    axes[0].plot(history.history['mse'], label='Train MSE={}'.format(round(history.history['mse'][-1], 3)))
    axes[0].plot(history.history['val_mse'], label='Validation MSE={}'.format(round(history.history['val_mse'][-1], 3)))
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('MEAN SQUARED ERROR', fontsize=12)
    axes[0].legend(loc="upper right")

    # plot scatter prediction and actual values
    axes[1].scatter(y_test, test_predictions, c='r')
    axes[1].set_xlabel('True Values')
    axes[1].set_ylabel('Predicted Values')
    axes[1].axis('equal')
    axes[1].axis('square')
    axes[1].set_xlim([0, plt.xlim()[1]])
    axes[1].set_ylim([0, plt.ylim()[1]])
    _ = axes[1].plot([-100, 100], [-100, 100])

    # To save the plot locally
    plt.savefig('tensorflow_keras_regression.png', bbox_inches='tight')
    plt.show()