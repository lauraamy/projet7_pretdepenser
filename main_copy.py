import joblib
import pickle
import re
import pandas as pd
import numpy as np
from fastapi import FastAPI
import lightgbm as lgb
import sklearn

app = FastAPI()

model = joblib.load('../Models/lgbm__trained_my_hypers_my_scorer1.sav')

# Define predict function
# @app.post('/predict')
def predict(credit_id):
    # data = pd.read_csv("MY_train_x.csv")
    # read the df obtained from merging the csv files and keeping the rows where TARGET is a missing value
    main_model_df = pd.read_csv('../my_csv_files/df_for_modelling.csv')
    main_model_df = main_model_df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    # credit_id between : 100002 - 456250
    row_of_interest = main_model_df[main_model_df['SK_ID_CURR'] == credit_id]
    # Create arrays and dataframes to store results
    feats = [f for f in main_model_df.columns if f not in ['TARGET', 'Unnamed: 0', 'Unnamed0',           
                                                           'SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV',
                                                           'index']]
    #probability = model.predict_proba(data.iloc[:, 1:]) 
    #prediction = model.predict(data.iloc[:, 1:]) 
    probability = model.predict_proba(row_of_interest[feats]) 
    prediction = model.predict(row_of_interest[feats]) 
    probability = pd.DataFrame(probability).astype(str)
    #.to_string()
    prediction = pd.DataFrame(prediction).astype(str)
    return {'Probability of 0 (non default) ' : probability[0][0], 
            'Probability of 1 (default) ' : probability[1][0],
            'Final Prediction ' : prediction[0][0]}

@app.get('/')
def get_root():
    return {'message': 'Welcome to Credit Home Prediction'}

@app.get('/credit_decision/{credit_id}')
async def get_credit_decision(credit_id: int):
    return predict(credit_id)

#if __name__ == '__main__':
#    uvicorn.run(app, host='127.0.0.1', port=8000)
# to run, make sure I am in the right folder (git) and make sure I am in the right virtual env
# command: uvicorn main_copy:app --reload