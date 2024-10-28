
path_python_material = ".." # REPLACE WITH YOUR PATH
model_id = "lr1"


# If unsure, print current directory path by executing the following in a new cell:
# !pwd

import uvicorn
import numpy as np
import pickle
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import logging
from pydantic import BaseModel
import shap
import sys
import pandas as pd
from typing import List
import datetime
from io import StringIO

sys.path.insert(0, '../')

app = FastAPI()

# Load the pipeline
with open(f"{path_python_material}/models/{model_id}-pipeline.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)

class Transaction(BaseModel):
    transaction_amount: float
    customer_age: int
    customer_balance: float

# Route to get feature importance
@app.get("/feature-importance")
def get_feature_importance():
    importance = loaded_pipeline[1].coef_[0].tolist()
    features = ['transaction_a  mount', 'customer_age', 'customer_balance']
    feature_importance = dict(zip(features, importance))
    return {"feature_importance": feature_importance}

# Route to predict new observations
@app.post("/predict/")
def predict_fraud(transaction: Transaction):

    data_point = np.array([[
        transaction.transaction_amount,
        transaction.customer_age,
        transaction.customer_balance
    ]])

    # Make predictions
    prediction = loaded_pipeline.predict(data_point)
    print(prediction)

    # Get probabilities for each class
    probabilities = loaded_pipeline.predict_proba(data_point)
    print(probabilities)
    confidence = probabilities[0].tolist()
    print(confidence)

    # Shap values
    path = f"{path_python_material}/data/2-intermediate/dsif11-X_train_scaled.npy"
    print(path)
    X_train_scaled = np.load(path)
    explainer = shap.LinearExplainer(loaded_pipeline[1], X_train_scaled)
    shap_values = explainer.shap_values(data_point)
    print("SHAP", shap_values.tolist())

    return {
        "fraud_prediction": int(prediction[0]),
        "confidence": confidence,
        "shap_values": shap_values.tolist(),
        "features": ['transaction_amount', 'customer_age', 'customer_balance']
        }

@app.post('/predict_automation')
def predict_automation(files_to_process:List[str]):

    from conf.conf import landing_path_input_data, landing_path_output_data

    print(f"Files to process (beginning): {files_to_process}")
    if '.DS_Store' in files_to_process:
        files_to_process.remove('.DS_Store')
        print(f"Files to process: {files_to_process}")

    input_data = pd.concat([pd.read_csv(landing_path_input_data + "/" + f) for f in files_to_process], ignore_index=True, sort=False)

    # # generate predictions
    input_data['pred_fraud'] = loaded_pipeline.predict(input_data)
    input_data['pred_proba_fraud'] = loaded_pipeline.predict_proba(input_data.drop(columns=['pred_fraud']))[:, 1]
    input_data['pred_proba_fraud'] = input_data['pred_proba_fraud'].apply(lambda x: round(x, 5))

    now = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    input_data.to_csv(landing_path_output_data + "/api_tagged_" + now + ".csv", index=False)
    return {
        "Predictions saved in " + landing_path_output_data + "/api_tagged_" + now + ".csv"
    }


# Endpoint for feature importance
@app.get("/feature-importance/")
def get_feature_importance():
    # Coefficients for logistic regression
    importance = loaded_pipeline[1].coef_[0]
    feature_names = ["transaction_amount", "customer_age", "customer_balance"]

    # Return feature importance as a dictionary
    feature_importance = dict(zip(feature_names, importance))
    return {"feature_importance": feature_importance}

# Set up logging
logging.basicConfig(filename="batch_scoring.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Endpoint for batch processing
@app.post("/batch_scoring/")
async def batch_scoring(csv_transactions: UploadFile = File(...)):
    try:
        # Read the uploaded file content (asynchronously)
        contents = await csv_transactions.read()  # Read file contents (this is non-blocking)

        # Convert the contents into a string format to process as a CSV
        df_transactions = pd.read_csv(StringIO(contents.decode('utf-8')))

        # Convert DataFrame to a list of dictionaries for Pydantic validation
        transactions_dict = df_transactions.to_dict(orient='records')

        # Validate each row against the Pydantic model
        validated_data: List[Transaction] = []
        for record in transactions_dict:
            try:
                validated_data.append(Transaction(**record))
            except ValidationError as e:
                logging.error(f"Validation error for record {record}: {e}")
                raise HTTPException(status_code=400, detail=f"Validation error: {e}")

        # Log that the input validation passed
        logging.info(f"Input validation passed for file: {csv_transactions.filename}")

        # Make predictions (replace this with your actual ML model)
        predictions = loaded_pipeline.predict(df_transactions)

        df_transactions["predictions"] = predictions

        # Convert DataFrame to a list of dictionaries
        result = df_transactions.to_dict(orient='records')
        # Return the result using JSONResponse to ensure correct content type
        return JSONResponse(content=result)

        # Return the predictions as JSON
        # return df_transactions.to_json(orient='records')


    except pd.errors.EmptyDataError:
        # Handle empty CSV file
        error_message = "The uploaded file is empty."
        logging.error(f"Error processing file: {csv_transactions.filename} - {error_message}")
        raise HTTPException(status_code=400, detail=error_message)

    except pd.errors.ParserError:
        # Handle CSV parsing errors
        error_message = "The uploaded file is not a valid CSV."
        logging.error(f"Error processing file: {csv_transactions.filename} - {error_message}")
        raise HTTPException(status_code=400, detail=error_message)

    except Exception as e:
        # Log the error with traceback
        logging.error(f"Error processing file: {csv_transactions.filename} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing the batch scoring: {str(e)}")

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
