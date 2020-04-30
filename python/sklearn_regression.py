__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: Scikit-Learn Regression [using PipeLine] (LinearRegression vs RandomForestRegressor) 
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, cross_val_predict


# Load data
def load_data():
    '''
    Load boston data
    :return: X, y:
    '''
    from sklearn.datasets import load_boston
    X, y = load_boston(return_X_y=True)
    return X, y


# Plot regression results
def plot_regression_results(axes, y_true, y_pred, title, scores):
    '''
    Scatter plot of the predicted vs true labeles
    :param axes:
    :param y_true:
    :param y_pred:
    :param title:
    :param scores:
    :return:
    '''
    axes.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r', linewidth=2)
    axes.scatter(y_true, y_pred, marker='o',alpha=0.4)
    axes.set_xlim([y_true.min(), y_true.max()])
    axes.set_ylim([y_true.min(), y_true.max()])
    axes.set_xlabel('Actual')
    axes.set_ylabel('Predicted')
    extra = plt.Rectangle((0, 0), 0, 0, fc="w", fill=False, edgecolor='none', linewidth=0)
    axes.set_title(title)
    axes.legend([extra], [scores], loc='upper left')
    return


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Load a sample data
    X, y = load_data()

    # Plot settings
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(17, 9))  # set figure shape and size
    fig.tight_layout(pad=3.0)  # set the spacing between subplots in Matplotlib

    # Method 1: LinearRegression()
    # ----------------------------
    # Select features according to the k highest scores
    anova_filter = SelectKBest(f_regression, k=10)  # f_regression: F-value between label/feature for regression tasks

    # Imputation transformer for completing missing values
    imputer = SimpleImputer(strategy='mean')

    # Transform features by scaling each feature to a given range
    scaler = StandardScaler()

    # Define a regressor
    reg_model = LinearRegression()

    # Chain all the estimators and transformers stages into a Pipeline estimator
    pipeline = make_pipeline(anova_filter, imputer, scaler, reg_model)

    # Fit data to the pipeline
    score = cross_validate(pipeline, X, y, scoring=['r2', 'neg_mean_absolute_error'], cv=5, n_jobs=-1, verbose=0)

    # Make predictions using the test data (perform regression on the test data)
    y_pred = cross_val_predict(pipeline, X, y, n_jobs=-1, verbose=0)

    # Plot regression results
    plot_regression_results(axes[0], y, y_pred, 'Linear Regression',
                            (r'$R^2-score={:.2f} \pm {:.2f}$' + '\n' + r'$Mean\ Absolute\ Error\ (MAE)={:.2f} \pm {:.2f}$')
                            .format(np.mean(score['test_r2']),np.std(score['test_r2']),
                                    -np.mean(score['test_neg_mean_absolute_error']),
                                    np.std(score['test_neg_mean_absolute_error'])))


    # Method 2: RandomForestRegressor()
    # ----------------------------
    # Imputation transformer for completing missing values
    imputer = SimpleImputer(strategy='mean')

    # Transform features by scaling each feature to a given range
    scaler = StandardScaler()

    # Define a regressor
    reg_model = RandomForestRegressor()

    # Chain all the estimators and transformers stages into a Pipeline estimator
    pipeline = make_pipeline(imputer, scaler, reg_model)

    # Fit data to the pipeline
    score = cross_validate(pipeline, X, y, scoring=['r2', 'neg_mean_absolute_error'], cv=5, n_jobs=-1, verbose=0)

    # Make predictions using the test data (perform regression on the test data)
    y_pred = cross_val_predict(pipeline, X, y, n_jobs=-1, verbose=0)

    # Plot regression results
    plot_regression_results(axes[1], y, y_pred, 'Random Forest Regressor',
                            (r'$R^2-score={:.2f} \pm {:.2f}$' + '\n' + r'$Mean\ Absolute\ Error\ (MAE)={:.2f} \pm {:.2f}$')
                            .format(np.mean(score['test_r2']),np.std(score['test_r2']),
                                    -np.mean(score['test_neg_mean_absolute_error']),
                                    np.std(score['test_neg_mean_absolute_error'])))

    # To save the plot locally
    plt.savefig('sklearn_regression.png', bbox_inches='tight')
    plt.show()
