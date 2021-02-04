import pandas as pd
import pandas_gbq as pgbq
import numpy as np
from scipy import stats
import joblib
from sklearn import metrics
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import datetime
import glob
import subprocess
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from google.cloud import bigquery

bq_client = bigquery.Client()

def load_and_aggregate(table_name):
    print('Starting Spark')
    #SparkContext.setSystemProperty('spark.executor.memory', '15g')
    #sc = SparkContext("local", "eval_merchant_embedding")
    spark = SparkSession.builder \
        .config("spark.jars.packages", "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.13.1-beta") \
        .config("spark.executor.memory", "1g") \
        .config("spark.driver.memory", "1g") \
        .config("spark.driver.maxResultSize", "0.5g") \
        .config("spark.dynamicAllocation.enabled ", True) \
        .appName("telco_churn") \
        .getOrCreate()

    print(spark.sparkContext.getConf().getAll())

    # Use the Cloud Storage bucket for temporary BigQuery export data used
    # by the connector.
    bucket = 'gs://{0}/{1}/tmp'.format(dataset, table_name)
    spark.conf.set('temporaryGcsBucket', bucket)

    # Read the data from BigQuery as a Spark Dataframe.
    spark_df = spark.read.format('bigquery') \
        .option('table', '{0}:{1}.{2}'.format(project_id, dataset, table_name)) \
        .load()
    print('Loaded {0} columns from {1}.{2}'.format(len(spark_df.columns), dataset, table_name))

    return spark_df


# Execute the script
start = datetime.datetime.now()
print(start)

# Use pyspark to load prepare inputs
train = load_and_aggregate('train')
test = load_and_aggregate('test')

seed = 7

# Define inputs and outputs
X_train = train[:,:-1]
X_test = test[:,-1]
y_train = train[:,:-1]
y_test = test[:,-1]

print(X_train[:5])
print(y_train[:5])

# Train model
xgb_model = xgb.XGBClassifier(n_estimators=1000,
                              max_depth=5,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              objective='multi:softprob',
                              nthread=-1,
                              random_state=seed)

xgb_model.fit(X_train, y_train)


y_pred = xgb_model.predict(X_val)

print(metrics.accuracy_score(y_val, y_pred))
print(metrics.confusion_matrix(y_val, y_pred))
print(metrics.classification_report(y_val, y_pred))

duration = datetime.datetime.now() - start
print(datetime.datetime.now())
print(duration)

with open('telco_churn.save', 'wb') as f:
    joblib.dump(xgb_model, f)

