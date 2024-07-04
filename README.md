# bank-challenge-ml-project

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/ced4f6b6-f0ce-473e-b0a0-186c7041d701)

### 1) The Challenge:
This is a project focused on building a solution to predict if a customer, in a bank, will perform a specific transaction in the future. Not only we built the model, but also deployed it with a REST API (using Docker) to be accessed in a Streamlit front-end in order to apply the prediction to new customers in the bank. Also, the performace of the models trained, along with all the data used, was stored in BigQuery. We were only informed the the prediction of this specific transaction was necessary, with no other business details being disclosed. In order to simulate a dynamic data creation, the model was trained on a part of the training data available, and afterwards, incrementally, the trained model was tested on new unseen data to test the performance and simulate the degradation of the model's quality with new data. Additionally, the model was retrained in order to improve the performance for future predictions.

### 2) The Data:
The data was obtained on Kaggle, more specifically in a challenge developed by Santander. The data is available in the link: https://www.kaggle.com/competitions/santander-value-prediction-challenge/data .
By entering this link, you can download the data. As you can see in this repository, the data is not available, and the reason is that the data's size was too big to store normally. If you download the data, add it to a folder named 'data'.

### 3) The First Data Ingestion:
The data was first obtained in csv format. But we wanted to build the project based on the cloud data storage, for which Google BigQuery was chosen. In order to do the first data ingestion, PySpark was used locally in order to load the data into BigQuery to be consumed afterwards. There were two csv files uploaded: a file named 'train' containing 200k rows and 202 columnns and one named 'test' containing 200k rows and 202 columnns. In the table 'train', each row represents a customer and, of the 202 columns: 1 is named 'target' and represents if the customer made the specific transaction (1) or not (0), 1 is named 'ID_code' and represents a identifier code for each customer and the other 200 columns represent each a feature, with no specific names. Both tables were loaded into a Dataset named 'db_bank_original_datasets'. The script that makes this ingestion is named initial_ingestion.py and is located in the folder src.
