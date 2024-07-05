# bank-challenge-ml-project

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/ced4f6b6-f0ce-473e-b0a0-186c7041d701)

### 1) The Challenge:
This is a project focused on building a solution to predict if a customer, in a bank, will perform a specific transaction in the future. Not only we built the model, but also deployed it with a REST API (using Docker) to be accessed in a Streamlit front-end in order to apply the prediction to new customers in the bank. Also, the performace of the models trained, along with all the data used, was stored in BigQuery. We were only informed the the prediction of this specific transaction was necessary, with no other business details being disclosed. In order to simulate a dynamic data creation, the model was trained on a part of the training data available, and afterwards, incrementally, the trained model was tested on new unseen data to test the performance and simulate the degradation of the model's quality with new data. Additionally, the model was retrained in order to improve the performance for future predictions.

A detailed scheme of the model's functionalities and tools used can be found in the image above.

### 2) The Data:
The data was obtained on Kaggle, more specifically in a challenge developed by Santander. The data is available in the link: https://www.kaggle.com/competitions/santander-value-prediction-challenge/data .
By entering this link, you can download the data. As you can see in this repository, the data is not available, and the reason is that the data's size was too big to store normally. If you download the data, add it to a folder named 'data'.

### 3) The First Data Ingestion:
The data was first obtained in csv format. But we wanted to build the project based on the cloud data storage, for which Google BigQuery was chosen. In order to do the first data ingestion, PySpark was used locally in order to load the data into BigQuery to be consumed afterwards. There were two csv files uploaded: a file named 'train' containing 200k rows and 202 columnns and one named 'test' containing 200k rows and 202 columnns. In the table 'train', each row represents a customer and, of the 202 columns: one is named 'target' and represents if the customer made the specific transaction (1) or not (0), one is named 'ID_code' and represents a identifier code for each customer and the other 200 columns represent each a feature, with no specific names. Both tables were loaded into a Dataset named 'db_bank_original_datasets'. The script that makes this ingestion is named initial_ingestion.py and is located in the folder src. Here, the 'test' table does not have the 'target' column!

### 4) EDA and Model Development:
The whole EDA process, along with the feature engineering process and model selection can be seen in details and commented in the notebook named eda.ipynb. In it, before starting the studies and development, the train data located in db_bank_original_datasets was split into two other tables, also named 'train' and 'test', and were then loaded into BigQuery in a Dataset now named 'db_bank_experiments'. The 'test' table will be used afterwards as new unseen data to test the trained models and retrain them, simulating new data coming into the system, letting us detect model's performance degradation.

So, the 'train' data, located in the 'db_bank_experiments' Dataset, was used to apply the whole initial modeling process, resulting in our first model version. Sklearn, xgboost and lightgbm were the main frameworks used to develop the models.

The first model built and also the other models obtained after other retraining processes on new data can be found, in pickle format, within the artifacts folder. As a form of version control, each model is differentiated according to the date that appears in the end of the file descrition, which represents the date when the model was trained.

### 5) ML Cycle Simulation: Testing & Retraining:
After developing the first model, some auxiliary modules were developed tu support a mudularized version of the system in production and placed within the folder names src. Modules such as data_processing.py (to preprocess the data when training the model and in production), extract_load_bigquery.py (to execute loading and extracting data processes on BigQuery in production and when training the model) and model_definition.py (that defines the model used and its hyperparameters and the threshold function for the binary classification prediction).

With these modules defined, the scripts model_build.py and testing_env.py were developed. model_build.py extracts the desired data from BigQuery, preprocesses it, trains the model and saves the new trained model in the artifacts folder. In this script, we can also insert new unseen data to train new versions of the model (using here the data from db_bank_experiments.test in parts.
