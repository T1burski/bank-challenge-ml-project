import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List
from src.model_definition import ModelFunctions
import os

# setting the fast api characteristics
tags_metadata = [{"name": "bank-transaction", "description": "Specific Bank Transaction Ocurrence"}]
app = FastAPI(title = "Bank Transaction API",
              description = "Specific Bank Transaction Ocurrence",
              version = "1.0",
              contact = {"name": "Artur"},
              openapi_tags = tags_metadata)


class PredictionInput(BaseModel):
    data: List[List[float]]


# ==========================
# Here, we specify the model we want to use that is located
# in the artifacts folder. For version control, we have the date
# of the training 
model_specification = "20240627"
# ==========================

pickle_path = os.path.join('artifacts', f"model_{model_specification}.pkl")

#loading the model in pickle format
def load_model():
    with open(pickle_path, 'rb') as model:
        clf_model = pickle.load(model)
    return clf_model

clf_model = load_model()

# setting get method for root
@app.get("/")
def message():
    text = "API for specific bank transaction ocurrence by customer"
    return text

# setting post method - predict using the model
@app.post("/predict", tags = ["Predict_Transaction_Ocurrence"])
async def predict(input_data: PredictionInput):

    X_data = np.array(input_data.data)
    
    y_pred = ModelFunctions().class_threshold(clf_model, X_data)

    # builds return dict
    response = {
                "Potential Transaction Ocurrence": y_pred.tolist(),
               }
    
    return response

# running the api using uvicorn web server interface
if __name__ == "__main__":
    uvicorn.run(app, host = "0.0.0.0", port = 3000)