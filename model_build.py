import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pickle
from src.data_preprocessing import DataPreprocessor
from src.extract_load_bigquery import BigQuery_DataOps
from src.model_definition import ModelFunctions
from datetime import datetime

data = BigQuery_DataOps(sub_db="experiments").extract_data(tb_id="train")

data_add = BigQuery_DataOps(sub_db="experiments").extract_data(tb_id="test")
#data_add = data_add.sort_values(by=["ID_code"], ascending=True).head(30000)

data = pd.concat([data, data_add])

processed_data = DataPreprocessor(data).select_columns(env="train")

X = processed_data.drop("target", axis=1)
y = processed_data["target"]

clf_production = ModelFunctions().train_model(X, y)

current_date = datetime.now()
formatted_date = current_date.strftime("%Y%m%d")

pickle_path = f"artifacts\model_{formatted_date}.pkl"

with open(pickle_path, 'wb') as model_file:
    pickle.dump(clf_production, model_file)