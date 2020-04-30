__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Scikit-Learn Classifiers
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, precision_recall_curve, f1_score


# Load data
def load_data():
    '''
    Load titanic data
    :return: dataframe:
    '''
    dataframe = sns.load_dataset('titanic')
    return dataframe


# Preprocessing data, split into training and test part
def preprocessing_data(df):
    '''
    Data preprocessing for titanic data
    :param df:
    :return: X_train, X_test, y_train, y_test
    '''
    # Drop useless and duplicate features
    useless_features = ['class', 'who', 'adult_male', 'deck', 'embark_town', 'alive'] # the duplicated features (e.g. class = pclass)
    df = df.drop(useless_features, axis=1)

    # Define categorical and numerical features
    features = df.columns
    categorical_features = ['pclass', 'sex', 'embarked', 'alone']
    numerical_features = list(set(features) - set(categorical_features) - set('survived'))   # ['survived'] is the label column

    # Fill the nan values with the column mean
    for c in numerical_features:
        df[c] = df[c].fillna(df[c].mean())

    df = df.dropna() # drop the nan values of categorical features

    # Label encoder
    labelencoder = LabelEncoder()
    for categorical_feature in categorical_features:
        df[categorical_feature] = labelencoder.fit_transform(df[categorical_feature])

    # OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    for categorical_feature in  categorical_features:
        # Passing bridge-types-cat column to encoder (label encoded values of bridge_types)
        enc_df = pd.DataFrame(enc.fit_transform(df[[categorical_feature]]).toarray()).add_suffix(categorical_feature)
        df = df.join(enc_df) # merge with main df bridge_df on key values

    # Transform features by scaling each feature to a given range
    scaler = StandardScaler()
    for numerical_feature in  numerical_features:
        df[numerical_feature] = pd.DataFrame(scaler.fit_transform(df[[numerical_feature]]))  # fit to data, then transform it

    # Drop the original categorical features, instead keep the one hot encoded versions
    df = df.drop(categorical_features, axis=1)
    df = df.dropna()  # drop the nan values of categorical features

    y = df.pop('survived').astype('int') # label column

    # Split data -> train and test
    X_train, X_test, y_train, y_test = train_test_split(df, y, train_size=0.8) # split data to 80% train and 20% test
    return X_train, X_test, y_train, y_test


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load a sample data
    df = load_data()

    # Preprocessing data and split into training and test part
    X_train, X_test, y_train, y_test = preprocessing_data(df)

    # Define different classifiers
    clf_svm = svm.SVC(kernel='rbf', probability=True)
    clf_knn = KNeighborsClassifier(n_neighbors= int(np.sqrt(len(y_train))))  # rule of thumb for the best number of k
    clf_lr = LogisticRegression()
    clf_dt = DecisionTreeClassifier(max_depth=4)
    clf_rf = RandomForestClassifier(n_jobs=2, max_depth=5, n_estimators=10)
    clf_ada = AdaBoostClassifier(n_estimators=10)
    clf_gb = GradientBoostingClassifier(n_estimators=10, max_depth=4)
    clf_gnb = GaussianNB()
    clf_bnb = BernoulliNB()
    clf_vc = VotingClassifier(estimators = [('lr', clf_lr), ('rf', clf_rf), ('gnb', clf_gnb)], voting = 'soft')

    # Create a list of defined classifiers
    clf_lst = [clf_svm, clf_knn, clf_lr, clf_dt, clf_rf, clf_ada, clf_gb, clf_gnb, clf_bnb, clf_vc]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Fit data to the models for classifiers comparison
    for clf in clf_lst:
        clf.fit(X_train, y_train)            # fit classifier according to the training data

        y_pred = clf.predict(X_test)         # perform classification on the test data
        y_probs = clf.predict_proba(X_test)  # return the probability estimates for the test data
        y_probs = y_probs[:, 1]              # keep probabilities for the positive outcome only


        # Compute ROC curve and area under ROC curve
        clf_auc = roc_auc_score(y_test, y_probs)          # calculate scores
        clf_fpr, clf_tpr, _ = roc_curve(y_test, y_probs)  # calculate roc curves

        # plot the roc curve for the model
        axes[0].plot(clf_fpr, clf_tpr, label='{}: AUC-ROC={}'.format(type(clf).__name__, round(clf_auc,3)))

        # Compute PR curve and area under PR curve
        clf_precision, clf_recall, _ = precision_recall_curve(y_test, y_probs)
        clf_auc = auc(clf_recall, clf_precision)

        # plot the precision-recall curves
        axes[1].plot(clf_recall, clf_precision, label='{}: AUC-PR={}'.format(type(clf).__name__, round(clf_auc, 3)))

        # Summarize scores
        clf_acc = accuracy_score(y_test, y_pred)
        clf_f1 = f1_score(y_test, y_pred)
        print('{:<30} ->         Accuracy={:.3f}       F1_score={:.3f}'.format(type(clf).__name__, round(clf_acc,3), round(clf_f1,3)))


    # Plot settings
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('AREA UNDER RECEIVER OPERATING CHARACTERISTIC (AUC-ROC)', fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('AREA UNDER PRECISION-RECALL CURVE (AUC-PR)', fontsize=12)
    axes[1].legend(loc="lower left")

    # To save the plot locally
    plt.savefig('sklearn_classification.png', bbox_inches='tight')
    plt.show()
