from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from google.oauth2 import service_account
import pandas as pd
import json

def extract_data():
    train_data = spark.read.option("header", True).csv("data/train.csv")
    test_data = spark.read.option("header", True).csv("data/test.csv")

    return [(train_data, 'train'), (test_data, 'test')]


def load_data(data_list, sub_db):

    db_id = 'db_bank_' + sub_db
    service_account_info = json.load(open('GBQ.json'))
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info)

    for tb_data, tb_name in data_list:
        if isinstance(tb_data, pd.DataFrame):
            tb_data.to_gbq(credentials = credentials, destination_table = db_id + "." + tb_name, if_exists = 'replace')
        else:
            tb_data_df = tb_data.toPandas()
            tb_data_df.to_gbq(credentials = credentials, destination_table = db_id + "." + tb_name, if_exists = 'replace')


if __name__ == "__main__":

    # Create a Spark session
    spark = SparkSession.builder \
    .appName("Bank") \
    .config("spark.executor.memory", "6g") \
    .config("spark.driver.memory", "6g") \
    .getOrCreate()

    load_data(extract_data(), 'original_datasets')