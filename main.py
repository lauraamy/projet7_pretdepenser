import joblib
import re
import pandas as pd
import numpy as np
from fastapi import FastAPI

app = FastAPI()

@app.get('/')
def get_root():
    return {'message': 'Welcome to Credit Home Prediction'}

model = joblib.load('gitfirstprojmodel.sav')

# Define predict function
@app.post('/predict')
def predict():
    data = pd.read_csv("MY_train_x.csv")
    probability = model.predict_proba(data.iloc[:, 1:]) 
    prediction = model.predict(data.iloc[:, 1:]) 
    return {'prediction': prediction,
            'probability': probability}

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)