__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: DNN Classification [using TensorFlow 1.x] 
"""
import os
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve


# Load data
def load_data():
    '''
    Load adult data from UCI Machine learning repository
    :return: pandas dataframe, numerical_features, categorical_features
    '''
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race',  'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    numerical_features = ["age", "capital-gain", "capital-loss", "hours-per-week"]
    categorical_features = ['workclass', 'education', 'marital-status',
                    'occupation', 'relationship', 'race', 'sex', 'native-country']
    useless_features = ['education-num', 'fnlwgt']

    dataframe = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                            header=None,
                            sep=",",
                            names=column_names)
    df = dataframe.drop(useless_features, axis=1) # return df after removing a useless and duplicated column

    # Create binary label from categorical label [Method 1]
    label = {' <=50K': 0, ' >50K': 1}
    df["label"] = [label[item] for item in df["income"]]  # create binary label

    # Create binary label from categorical label [Method 2]
    #df["label"] = df["income"].apply(label_creator)

    df = df.drop("income", axis=1) # drop categorial label (keep only binary label)
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


# Preprocessing:  Preprocessing data, split into training and test part
def preprocessing(df, numerical_features):
    '''
    Preprocessing data, split into training and test part
    :param df:
    :param numerical_features:
    :return:
    '''
    # Fill the nan values with the column mean (numericalfeatures)
    for col in numerical_features:
        df[col] = df[col].fillna(df[col].mean())

    df = df.replace('?', np.nan) # replace values given in to_replace with value
    df = df.dropna() # drop the nan values of categorical features
    X = df.drop('label', axis=1) # features
    y = df['label'] # label
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    return X_train, X_test, y_train, y_test


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load data
    df, numerical_features, categorical_features = load_data()

    # Preprocessing data and split into training and test part
    X_train, X_test, y_train, y_test = preprocessing(df, numerical_features)

    # TensorFlow: Model initialization
    num_classes = 2  # number of classes
    batch_size = 100  # number of instances to be read each time (i.e. each iteration)
    iter_number = 5000 # number of steps in training stage
    hidden_layers = [20, 20] # number of units per each hidden layer
    model_path = os.getcwd() + '/saved_model'  # to save trained model in the current directory

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
    input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=batch_size, shuffle=True)

    # TensorFlow: Create the DNN classifier
    model = tf.estimator.DNNClassifier(feature_columns=feat_cols, #all the feature columns used by the model
                                       hidden_units=hidden_layers, # iterable of number hidden units per layer
                                       n_classes=num_classes, # number of label classes(default: 2)
                                       optimizer='Adam', # optimizers are:('Adagrad', 'Adam', 'Ftrl', 'RMSProp', 'SGD')
                                       activation_fn=tf.nn.relu, # activation function applied to each layer
                                       dropout=None, # the probability to drop out a given coordinate
                                       model_dir=model_path)

    # TensorFlow: Train the model
    model.train(input_fn=input_func, steps=iter_number)

    # TensorFlow: Evaluation the model
    pred_func = tf.estimator.inputs.pandas_input_fn(x=X_test, shuffle=False) # the prediction input
    predictions = model.predict(input_fn=pred_func) # predictions: a dict of class_ids, classes, ..., probabilities
    predictions = list(predictions) # convert the class type <generator> to list

    # Create a list of only class_ids of predictions
    y_pred = []
    for prediction in predictions:
        y_pred.append(prediction['class_ids'][0])

    # Summarize result
    print(classification_report(y_test, y_pred))

    # Plot settings
    fig = plt.subplots(nrows=1, ncols=1, figsize=(9, 9))  # set figure shape and size

    # Plot confusion matrix
    sns.set(font_scale=1.2)  # for label size
    color_map = sns.cubehelix_palette(dark=0, light=0.95, as_cmap=True)  # color_map for seaborn plot
    cm = confusion_matrix(y_test, y_pred) # confusion matrix
    sns.heatmap(cm, cmap=color_map, annot=True, annot_kws={"size": 12}, fmt="d")  # plot confusion matrix heatmap
    plt.title('CONFUSION MATRIX (HEATMAP)', fontsize=12)

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_dnnclassification_cm.png', bbox_inches='tight')
    plt.show()

    # False Positive Rate, True Positive Rate and Area Under Curve-ROC
    fpr, tpr, roc_auc  = dict(), dict(), dict()
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    # Precision, Recall and Area Under Curve-PR
    pr, re, pr_auc  = dict(), dict(), dict()
    pr, re, _ = precision_recall_curve(y_test, y_pred)
    pr_auc = auc(re, pr)

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # plot the roc curve for the model
    axes[0].plot(fpr, tpr, label='{}: AUC-ROC={}'.format(type(model).__name__, round(roc_auc,3)))
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('AREA UNDER RECEIVER OPERATING CHARACTERISTIC (AUC-ROC)', fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].plot([0, 1], [0, 1], color='k', alpha=0.3, lw=2, linestyle='--')

    # plot the precision-recall curve for the model
    axes[1].plot(re, pr, label='{}: AUC-PR={}'.format(type(model).__name__, round(pr_auc, 3)), color='r')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('AREA UNDER PRECISION-RECALL CURVE (AUC-PR)', fontsize=12)
    axes[1].legend(loc="lower left")

    # To save the plot locally
    plt.savefig('tensorflow_v1.x_dnnclassification.png', bbox_inches='tight')
    plt.show()