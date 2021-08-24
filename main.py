import joblib
import pickle
import re
import pandas as pd
import numpy as np
from fastapi import FastAPI
import lightgbm as lgb
import sklearn

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to Credit Home Prediction'}

model = joblib.load('lgbm__trained_my_hypers_my_scorer1.sav')

# Define predict function
@app.post('/predict')
def predict():
    data = pd.read_csv("MY_train_x.csv")
    probability = model.predict_proba(data.iloc[:, 1:]) 
    prediction = model.predict(data.iloc[:, 1:]) 
    probability = pd.DataFrame(probability).to_string()
    prediction = pd.DataFrame(prediction).to_string()
    return {'probability' : probability,
            'prediction' : prediction}

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)