import streamlit as st
import requests
import numpy as np
import pandas as pd
from src.extract_load_bigquery import BigQuery_DataOps
from src.data_preprocessing import DataPreprocessor


st.title("Prediction of Bank Transaction Execution by Clients: An ML Web App")
st.write("""### This is a simulator of a production environment in which a classification model predicts if a client will do a specific bank transaction or not""")
st.divider()

n_clients = st.number_input('Select the number of most recent new clients you wish to predict if they will likely execute the transaction:', min_value=1, step=1)
data = BigQuery_DataOps(sub_db="original_datasets").extract_data(tb_id="test", prod=True, n_clients=n_clients)
st.divider()

data_processed = DataPreprocessor(data).select_columns(env="prod")

payload = {"data" : data_processed.values.tolist()}
response = requests.post("http://localhost:3000/predict", json = payload).json()


clients = data["ID_code"]
predictions = np.where(np.array(response["Potential Transaction Ocurrence"])==1, "Yes", "No")
binary_response = response["Potential Transaction Ocurrence"]

final_df = pd.DataFrame({"Client": clients,
                         "Potential Executor of Transaction": predictions}).reset_index(drop=True)

st.write("""### Output: List of clients and if they are likely to execute the transaction in the future""")
st.write(final_df)