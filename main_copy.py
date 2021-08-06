import joblib
import pickle
import re
import pandas as pd
import numpy as np
from fastapi import FastAPI
import lightgbm as lgb
import sklearn

app = FastAPI()

model = joblib.load('lgbm__trained_my_hypers_my_scorer1.sav')

# Define predict function
#@app.post('/predict')
def predict(credit_id):
    #data = pd.read_csv("MY_train_x.csv")
    main_model_df = pd.read_csv('df_for_modelling.csv')
    main_model_df = main_model_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    row_of_interest = main_model_df[main_model_df['SK_ID_CURR'] == credit_id]
    # Create arrays and dataframes to store results
    feats = [f for f in main_model_df.columns if f not in ['TARGET', 'Unnamed: 0', 'Unnamed0',           
                                                           'SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV',
                                                           'index']]
    #probability = model.predict_proba(data.iloc[:, 1:]) 
    #prediction = model.predict(data.iloc[:, 1:]) 
    probability = model.predict_proba(row_of_interest[feats]) 
    prediction = model.predict(row_of_interest[feats]) 
    probability = pd.DataFrame(probability).to_string()
    prediction = pd.DataFrame(prediction).to_string()
    return {'probability' : probability,
            'prediction' : prediction}

@app.get('/')
def get_root():
    return {'message': 'Welcome to Credit Home Prediction'}

@app.get('/credit_decision/{credit_id}')
async def get_credit_decision(credit_id: int):
    return predict(credit_id)

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)