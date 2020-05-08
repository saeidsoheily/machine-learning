__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: TensorFlow Keras Binary Classification [using TensorFlow 2.x-Keras] 
"""
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# Load data
def load_data():
    '''
    Load adult data from UCI Machine learning repository
    :return: pandas dataframe, numerical_features, categorical_features
    '''
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    numerical_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ['workclass', 'education', 'marital-status',
                            'occupation', 'relationship', 'race', 'sex', 'native-country']
    useless_features = ['education-num', 'fnlwgt']

    dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                            header=None,
                            sep=",",
                            names=column_names)
    df = dataframe.drop(useless_features, axis=1)  # return df after removing a useless and duplicated column

    # Create binary label from categorical label [Method 1]
    label = {' <=50K': 0, ' >50K': 1}
    df["label"] = [label[item] for item in df["income"]]  # create binary label

    # Create binary label from categorical label [Method 2]
    # df["label"] = df["income"].apply(label_creator)

    df = df.drop("income", axis=1)  # drop categorial label (keep only binary label)
    return df, numerical_features, categorical_features


# Preprocessing: Create binary label
def label_creator(label):
    '''
    Create binary label from categorical label
    :param label: categorical label
    :return:
    '''
    if label == ' >50K':
        return 1
    else:
        return 0


# Preprocessing:  Preprocessing data, split into training, validation and test part
def preprocessing(df, numerical_features):
    '''
    Preprocessing data, split into training, validation and test part
    :param df:
    :param numerical_features:
    :return: X_train, X_val, X_test, y_train, y_val, y_test
    '''
    # Fill the nan values with the column mean (numericalfeatures)
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].mean())

    df = df.replace(' ?', np.nan)  # replace values given in to_replace with value
    df = df.dropna()  # drop the nan values of categorical features
    X = df.drop('label', axis=1)  # features
    y = df['label']  # label

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
    df, numerical_features, categorical_features = load_data()

    # Preprocessing data and split into training, validation and test part
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessing(df, numerical_features)

    # TensorFlow: Model initialization
    label = 'label'  # label column name
    num_classes = 2  # number of classes
    buffer_size = X_train.shape[0]  # shuffle buffer size
    batch_size = 128  # number of instances to be read each time (i.e. each iteration)
    iter_number = 200  # number of steps in training stage

    # TensorFlow: Create the TF feature columns for numerical features
    age = tf.feature_column.numeric_column('age')
    capital_gain = tf.feature_column.numeric_column('capital-gain')
    capital_loss = tf.feature_column.numeric_column('capital-loss')
    hours_per_week = tf.feature_column.numeric_column('hours-per-week')

    # TensorFlow: Create the TF feature columns for categorical features
    workclass = tf.feature_column.categorical_column_with_hash_bucket('workclass', hash_bucket_size=10)
    education = tf.feature_column.categorical_column_with_hash_bucket('education', hash_bucket_size=10)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket('marital-status', hash_bucket_size=10)
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=10)
    relationship = tf.feature_column.categorical_column_with_hash_bucket('relationship', hash_bucket_size=10)
    race = tf.feature_column.categorical_column_with_hash_bucket('race', hash_bucket_size=10)
    sex = tf.feature_column.categorical_column_with_vocabulary_list('sex', [' Female', ' Male'])
    native_country = tf.feature_column.categorical_column_with_hash_bucket('native-country', hash_bucket_size=10)

    # TensorFlow: Create feature columns by considering both categorical and numerical features [Method 1]
    feat_cols = [age, capital_gain, capital_loss, hours_per_week,
                 tf.feature_column.embedding_column(workclass, dimension=10),
                 tf.feature_column.embedding_column(education, dimension=10),
                 tf.feature_column.embedding_column(marital_status, dimension=10),
                 tf.feature_column.embedding_column(occupation, dimension=10),
                 tf.feature_column.embedding_column(relationship, dimension=10),
                 tf.feature_column.embedding_column(race, dimension=10),
                 tf.feature_column.indicator_column(sex),
                 tf.feature_column.embedding_column(native_country, dimension=10)
                 ]

    # TensorFlow: Create the input function
    train_ds = df_to_dataset(X_train, y_train, batch_size=batch_size)
    val_ds = df_to_dataset(X_val, y_val, shuffle=False)
    test_ds = df_to_dataset(X_test, y_test, shuffle=False)

    # TensorFlow: Create the model
    model = tf.keras.Sequential([
        tf.keras.layers.DenseFeatures(feat_cols),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # regularization, useful for neural networks
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),  # regularization, useful for neural networks
        tf.keras.layers.Dense(1)])  # keras uses sigmoid on the output layer

    # TensorFlow: Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics='accuracy')

    # TensorFlow: Train the model
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', # quantity to be monitored (validation loss)
                                                      verbose=1,  # verbosity mode
                                                      patience=20,  # epochs with no improvement to be stopped
                                                      mode='min',  # ("auto", "min", "max")
                                                      restore_best_weights=True)  # restore weights from the best value

    history = model.fit(train_ds,  # (train_features, train_label)
                        validation_data=val_ds,  # (val_features, val_label)
                        callbacks=[early_stopping],  # stop training when a monitored quantity has stopped improving
                        epochs=iter_number)

    # TensorFlow: Save the model
    model.save('tf_keras_classification.model')

    # TensorFlow: Restore the model (uncomment to restore the saved model)
    #model = tf.keras.models.load_model('tf_keras_classification.model')

    # TensorFlow: Evaluation the model (test set)
    loss, accuracy = model.evaluate(test_ds) # (test_features, test_label)
    test_predictions = model.predict(test_ds)
    test_predictions = tf.keras.activations.sigmoid(test_predictions)  # apply sigmoid function
    y_pred = (test_predictions.numpy() > 0.5).astype(int)  # convert predicted probabilities to labels
    cm = confusion_matrix(y_test, y_pred)  # confusion matrix

    # Summarize result
    print("Model Accuracy on Test Data: ", accuracy)

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the accuracy curve for the model
    axes[0].plot(history.history['accuracy'],
                 label='Train Accuracy={}'.format(round(history.history['accuracy'][-1], 3)))
    axes[0].plot(history.history['val_accuracy'],
                 label='Validation Accuracy={}'.format(round(history.history['val_accuracy'][-1], 3)))
    axes[0].set_xlim([0.0, len(history.history['accuracy'])])
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('MODEL ACCURACY', fontsize=12)
    axes[0].legend(loc="lower right")

    # plot the confusion matrix for the model
    sns.set(font_scale=1.2)  # for label size
    color_map = sns.cubehelix_palette(dark=0, light=0.95, as_cmap=True)  # color_map for seaborn plot
    sns.heatmap(cm, cmap=color_map, annot=True, annot_kws={"size": 12}, fmt="d")  # plot confusion matrix heatmap
    axes[1].set_title('CONFUSION MATRIX (HEATMAP)', fontsize=12)

    # To save the plot locally
    plt.savefig('tensorflow_keras_classification.png', bbox_inches='tight')
    plt.show()
