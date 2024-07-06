# bank-challenge-ml-project

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/ced4f6b6-f0ce-473e-b0a0-186c7041d701)

### 1) The Challenge:
This is a project focused on building a solution to predict if a customer, in a bank, will perform a specific transaction in the future. Not only we built the model, but also deployed it with a REST API (using Docker) to be accessed in a Streamlit front-end in order to apply the prediction to new customers in the bank. Also, the performace of the models trained, along with all the data used, was stored in BigQuery. We were only informed the the prediction of this specific transaction was necessary, with no other business details being disclosed. In order to simulate a dynamic data creation, the model was trained on a part of the training data available, and afterwards, incrementally, the trained model was tested on new unseen data to test the performance and simulate the degradation of the model's quality with new data. Additionally, the model was retrained in order to improve the performance for future predictions.

A detailed scheme of the model's functionalities and tools used can be found in the image above.

### 2) The Data:
The data was obtained on Kaggle, more specifically in a challenge developed by Santander. The data is available in the link: https://www.kaggle.com/competitions/santander-customer-transaction-prediction/data .
By entering this link, you can download the data. As you can see in this repository, the data is not available, and the reason is that the data's size was too big to store normally. If you download the data, add it to a folder named 'data'.

### 3) The First Data Ingestion:
The data was first obtained in csv format. But we wanted to build the project based on the cloud data storage, for which Google BigQuery was chosen. In order to do the first data ingestion, PySpark was used locally in order to load the data into BigQuery to be consumed afterwards. There were two csv files uploaded: a file named 'train' containing 200k rows and 202 columnns and one named 'test' containing 200k rows and 202 columnns. In the table 'train', each row represents a customer and, of the 202 columns: one is named 'target' and represents if the customer made the specific transaction (1) or not (0), one is named 'ID_code' and represents a identifier code for each customer and the other 200 columns represent each a feature, with no specific names. Both tables were loaded into a Dataset named 'db_bank_original_datasets'. The script that makes this ingestion is named initial_ingestion.py and is located in the folder src. Here, the 'test' table does not have the 'target' column!

### 4) EDA and Model Development:
The whole EDA process, along with the feature engineering process and model selection can be seen in details and commented in the notebook named eda.ipynb. In it, before starting the studies and development, the train data located in db_bank_original_datasets was split into two other tables, also named 'train' and 'test', and were then loaded into BigQuery in a Dataset now named 'db_bank_experiments'. The 'test' table will be used afterwards as new unseen data to test the trained models and retrain them, simulating new data coming into the system, letting us detect model's performance degradation.

So, the 'train' data, located in the 'db_bank_experiments' Dataset, was used to apply the whole initial modeling process, resulting in our first model version. Sklearn, xgboost and lightgbm were the main frameworks used to develop the models.

The first model built and also the other models obtained after other retraining processes on new data can be found, in pickle format, within the artifacts folder. As a form of version control, each model is differentiated according to the date that appears in the end of the file descrition, which represents the date when the model was trained.

### 5) ML Cycle Simulation: Testing & Retraining:
After developing the first model, some auxiliary modules were developed to support a mudularized version of the system in production and placed within the folder names src. Modules such as data_processing.py (to preprocess the data when training the model and in production), extract_load_bigquery.py (to execute loading and extracting data processes on BigQuery in production and when training the model) and model_definition.py (that defines the model used and its hyperparameters and the threshold function for the binary classification prediction).

With these modules defined, the scripts model_build.py and testing_env.py were developed. model_build.py extracts the desired data from BigQuery, preprocesses it, trains the model and saves the new trained model in the artifacts folder. In this script, we can also insert new unseen data to train new versions of the model (using here the data from db_bank_experiments.test in parts.

In order to simulate the creation of new data, which is followed by testing the model on new data to gather information about the performance of the model currently in production and possibly retraining the model using new data, the process describedby the image below was created, in which new models were trained on new data after we verified that the model's performance got worse over time:

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/aa0bbfe1-3c6f-4634-8047-c0d4bd32eb6d)

head and tail were simple table operations to select different data in the db_bank_experiments.test table, simulating the creation of new data over time. The db_bank_experiments.test table has 60k rows.

In the image below, on the left, we can see how our BigQuery DW was in the end of the project, with every dataset and table organized. Also, on the right, we compare the performance of the models on the unseen data db_bank_experiments.test.tail(30000) Before and After including db_bank_experiments.test.head(30000) in the training data to create the model. We can seethat indeed the performance improved when we added more data to train the model! For example, the ROC_AUC improved from 0.902 to 0.972 and the F1-Score (for the '1' positive class) improved from 0.588 to 0.755. 

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/05918f94-49ae-4d49-b7c4-fac399dc8c6e)

The final model (that is running in production) in the model model_20240627.pkl, which was trained with all db_bank_experiments data.

### 6) Deploying the Model Using FastAPI and Docker:
After the final model was trained, we chose to deply it using a REST API, building it with FastAPI and putting it in a Docker container. The file app.py contains the REST API, which runs using the ASGI web server uvicorn.

### 7) Streamlit Front-End:
The interaction with the final user was decided to be made in a Web App using Streamlit in which the user (the bank employee that was assigned to analyze various information about new customers, such as if they are going to, in the future, execute the specfic bank transaction). When the user opens the app, the following screen appears:

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/128cdc5d-79bd-4c5c-8389-eb54f449710f)

For now, the app allows the user to select the number of most recent customers the wish to predict if will make, in the future, the specific bank transaction. When the number of customers is chosen, the app will run the REST API (which is running in the Docker container), and a table below will appear containing the customer ID and the respective prediction, what can be seen in the example below in which the user chose 20 new customers to apply the predictions on:

![image](https://github.com/T1burski/bank-challenge-ml-project/assets/100734219/494f60d7-aabc-4619-94b4-4ce5d755d235)

Everytime the user changes the number, the classification will run again. Internally, everytime a number is chosen, the backend extracts the data from BigQuery and applies the Model through the FastAPI implementation using the post method. All customers that are pulled here are from the table db_bank_original_datasets.test, in which we do not have the target column.

### 8) How to Run the Project:
-- Create a local folder and open a terminal in it

-- Clone this repository using the command:

`git clone https://github.com/T1burski/bank-challenge-ml-project.git`

-- On the terminal, run the command:

`poetry install`

and then the command:

`poetry shell`

-- Now, we need to create the Docker image and then build the container using it. For this, first run the following command:

`docker build -t bank-transaction:api .`

and then the command:

`docker run -dit --name bank-transaction-api -p 3000:3000 bank-transaction:api `

-- After this, with our API running in the container, let's start the streamlit front-end with the command:

`streamlit run front.py `

And then your browser would be able to run the Web App!

Obs: For this to work, you need to set up your BigQuery Account (Sandbox one, for example), create the service account key, save it to a json file, put it within the root of the repo and name it GBQ.json. Also, you will need to make the initial ingestions of the data and run the commands of the eda.ipynb to create all datasets and tables on the BigQuery cloud.

