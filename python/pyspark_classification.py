__author__ = 'Saeid SOHILY-KHAH'
"""
Machine learning algorithms: PySpark Classification (RandomForestClassifier)
"""
import os
import pyspark
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler, IndexToString
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from sklearn.metrics import roc_curve, auc, precision_recall_curve


# System settings
'''
os.environ['SPARK_HOME']="<SPARK HOME DIR.>"
os.environ['PYSPARK_PYTHON'] = "python3"
os.environ['PYSPARK_DRIVER_PYTHON'] = "python3"
os.environ['SPARK_LOCAL_IP'] = "<IP ADDRESS>"
os.environ['JAVA_HOME'] = "<JAVA HOME DIR.>" # '/usr/lib/jvm/java-8-openjdk-amd64'
'''


# Load data
def load_data():
    '''
    Load adult data from UCI Machine learning repository
    :return: spark dataframe, categorical_features, numerical_features, label, label_distinct_values
    '''
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                    'occupation', 'relationship', 'race',  'sex', 'capital-gain', 'capital-loss',
                    'hours-per-week', 'native-country', 'income']
    dataframe = spark.createDataFrame(
        pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
                    header=None, sep=","),
        schema=column_names)

    df = dataframe.drop('education-num') # return df after removing a duplicated column
    categorical_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    numerical_features = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week"]
    label = "income"
    label_distinct_values = df.select(label).distinct().rdd.flatMap(lambda x: x).collect() # list of distinct labels
    label_distinct_values = sorted([item.strip() for item in label_distinct_values])
    return df, categorical_features, numerical_features, label, label_distinct_values


# Preprocessing data, split into training and test part
def preprocessing_data(df, numerical_features):
    '''
    Prepocessing data and split into train and test
    :param df: dataframe
    :param numerical_features: list of numericalfeatures
    :return: df_train, df_test
    '''
    # Cast numerical features' type to Double
    for numerical_feature in numerical_features:
        df = df.withColumn(numerical_feature, df[numerical_feature].cast(DoubleType()))

    # Split the data into training and test sets (20% held out for testing)
    df_rep = df.repartition(100, df.schema.names[0]) # to overcome randomSplit inconsistent behavior
    (df_train, df_test) = df_rep.randomSplit([0.8, 0.2])
    return df_train, df_test


# ------------------------------------------------ MAIN ----------------------------------------------------
if __name__ == '__main__':
    # Spark session
    spark = (pyspark.sql.SparkSession
             .builder  # this method is created for constructing a SparkSession
             .appName("PySpark Classification")  # a name for the application which will be shown in the spark Web UI
             .master('local[*]')  # spark master: “local[*]”: run locally with all cores, “spark://master:7077”: run on a spark standalone cluster
             .config('spark.executor.memory', '1g')
             .config('spark.executor.cores', '2')
             .config('spark.cores.max', '2')
             .config('spark.driver.memory', '1g')
             .getOrCreate())  # gets an existing SparkSession or, if not exist, creates a new SparkSession

    # Load a sample data
    df, categorical_features, numerical_features, label, label_distinct_values = load_data()

    # Preprocessing data and split into training and test part
    df_train, df_test = preprocessing_data(df, numerical_features)

    # Category indexing with StringIndexer
    stringIndexer = [StringIndexer(inputCol=c,
                                   outputCol=c + "Index").setHandleInvalid('keep') for c in categorical_features]

    # Convert categorical features into binary SparseVectors
    encoder = OneHotEncoderEstimator(inputCols=[c + 'Index' for c in categorical_features],
                                     outputCols=[c + 'ClassVec' for c in categorical_features])

    # Transform all features into a vector using VectorAssembler
    features = [c + "ClassVec" for c in categorical_features] + numerical_features
    vectorAssembler = VectorAssembler(inputCols=features, outputCol="features")

    # Convert label into label indices using the StringIndexer
    label_stringIndexer = StringIndexer(inputCol=label, outputCol="labelIndex") # encode labels to label indices

    # Define classifier
    clf = RandomForestClassifier(featuresCol="features", labelCol="labelIndex")

    # Convert indexed labels back to original labels
    labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel", labels=label_distinct_values)

    # Chain all the estimators and transformers stages into a Pipeline estimator
    pipeline = Pipeline(stages = stringIndexer + [encoder, vectorAssembler, label_stringIndexer, clf, labelConverter])

    # Patameter grid builder
    clfParamGrid = ParamGridBuilder().addGrid(clf.numTrees, [50, 150]).addGrid(clf.maxDepth, [5, 10]).build()

    # Cross validation
    crossValidator = CrossValidator(estimator=pipeline,
                                    estimatorParamMaps=clfParamGrid,
                                    evaluator=BinaryClassificationEvaluator(labelCol='labelIndex'),
                                    numFolds=5)

    # Fit data to get a compiled pipeline with transformers and fitted models
    model = crossValidator.fit(df_train)

    # Make predictions on test data using the transform() method.
    predictions = model.transform(df_test)  # transform() will only use the 'features' column

    # Select example rows of predictions to display
    predictions.select("features", "labelIndex", "prediction", "probability", "predictedLabel", label).show(20)

    # Summarize scores
    evaluator_ac = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",
                                                     metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",
                                                     metricName="f1")
    evaluator_wp = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",
                                                     metricName="weightedPrecision")
    evaluator_wr = MulticlassClassificationEvaluator(labelCol="labelIndex", predictionCol="prediction",
                                                     metricName="weightedRecall")
    ac = evaluator_ac.evaluate(predictions)
    f1 = evaluator_f1.evaluate(predictions)
    wp = evaluator_wp.evaluate(predictions)
    wr = evaluator_wr.evaluate(predictions)
    print('{:<25} ->    Accuracy={:.3f}'.format(type(clf).__name__, round(ac, 3)))
    print('{:<25} ->    F1-score={:.3f}'.format(type(clf).__name__, round(f1, 3)))
    print('{:<25} ->    Weighted Precision={:.3f}'.format(type(clf).__name__, round(wp, 3)))
    print('{:<25} ->    Weighted Recall={:.3f}'.format(type(clf).__name__, round(wr, 3)))

    # plot the roc curve for the model
    results = predictions.select(['probability', 'labelIndex']).collect()
    results_lst = [(float(y_[0][0]), 1.0 - float(y_[1])) for y_ in results]
    y_test = [y_[1] for y_ in results_lst]
    y_pred = [y_[0] for y_ in results_lst]

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
    axes[0].plot(fpr, tpr, label='{}: AUC-ROC={}'.format(type(clf).__name__, round(roc_auc,3)))
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('AREA UNDER RECEIVER OPERATING CHARACTERISTIC (AUC-ROC)', fontsize=12)
    axes[0].legend(loc="lower right")
    axes[0].plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')

    # plot the precision-recall curve for the model
    axes[1].plot(re, pr, label='{}: AUC-PR={}'.format(type(clf).__name__, round(pr_auc, 3)))
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('AREA UNDER PRECISION-RECALL CURVE (AUC-PR)', fontsize=12)
    axes[1].legend(loc="lower left")

    # To save the plot locally
    plt.savefig('pyspark_classification.png', bbox_inches='tight')
    plt.show()

    spark.stop() # close the current session

