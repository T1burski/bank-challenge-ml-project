import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import classification_report, roc_auc_score
import pickle
from src.data_preprocessing import DataPreprocessor
from src.extract_load_bigquery import BigQuery_DataOps
from src.model_definition import ModelFunctions
from datetime import datetime


'''
This script has the objective to test the desired model in the artifacts folder
on new data, saving to Google BQ the performance metrics for later comparisons.

For our MLOps simulations, we used head and tail dataframes manipulations to select
various customers in the testing data to simulate the creation of new 
data to test on and check if there was any degradation in the model's performace.
See the commented lines: the first selects different customers from time to time
and the second one customizes the name of the table outputed with the performance
metrics according to the model used, tail or head and date of the test. 
'''


model_specification = "20240625"
pickle_path = f"artifacts\model_{model_specification}.pkl"

def load_model():
    with open(pickle_path, 'rb') as model:
        clf_model = pickle.load(model)
    return clf_model
clf_prod = load_model()

data = BigQuery_DataOps(sub_db="experiments").extract_data(tb_id="test")

#data = data.sort_values(by=["ID_code"], ascending=True).head(30000)
data = data.sort_values(by=["ID_code"], ascending=True).tail(30000)

processed_data = DataPreprocessor(data).select_columns(env="train")

X = processed_data.drop("target", axis=1)
y = processed_data["target"]

y_pred = ModelFunctions().class_threshold(clf_prod, X)

current_date = datetime.now()
formatted_date = current_date.strftime("%Y%m%d")

report = classification_report(y.values.astype('int'), y_pred, output_dict=True)
report_df = pd.DataFrame.from_dict(report).transpose()
report_df = report_df.reset_index()
report_df.rename(columns={'index': 'Metric'}, inplace=True)

report_df['roc_auc'] = roc_auc_score(y.values.astype('int'), clf_prod.predict_proba(X.values.astype('float'))[:, 1])
report_df['test_date'] = formatted_date
model_name = f"model_{model_specification}"
report_df['model'] = model_name

#table_output_name = f"test_metrics_head_{model_name}_d{formatted_date}"
table_output_name = f"test_metrics_tail_{model_name}_d{formatted_date}"

BigQuery_DataOps(sub_db="metrics").load_data(tb_data=report_df, tb_name=table_output_name)