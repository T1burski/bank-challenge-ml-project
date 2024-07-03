# Import
import requests
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.extract_load_bigquery import BigQuery_DataOps
from src.data_preprocessing import DataPreprocessor


'''
This script was built to test the api the same way
a client (in our case the streamlit front-end) does it
'''

data = BigQuery_DataOps(sub_db="experiments").extract_data(tb_id="test")
data = data.sort_values(by=["ID_code"], ascending=False).head(5)
data = DataPreprocessor(data).select_columns(env="prod")

# Dados que serão passados para a API
payload = {"data" : data.values.tolist()}

# Faz a requisição à API
response = requests.post("http://localhost:3000/predict", json = payload).json()

print("\n")
print("\n")
print("\n")
print("final output:")
print(response)