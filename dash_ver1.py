# Loading packages #
import streamlit as st
import streamlit.components.v1 as components
import requests,json

import pandas as pd
import numpy as np

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Title",
    layout="wide",
    initial_sidebar_state="expanded",
)


# read the df obtained from merging the csv files and keeping the rows where TARGET is a missing value
# Loading of the data
@st.cache
def load_data():
    data = pd.read_csv('../my_csv_files/df_for_modelling.csv')
    return data

df = load_data()




if __name__ == "__main__":
    
    st.info("Info Text")

    with st.sidebar:
        st.title("Title Text")
        # Sidebar creation ####################################################################
        st.sidebar.title('Sidebar Parameters Text ')
        st.sidebar.markdown("Side bar markdown Text:")

        #cols = ["EXT_SOURCE_2", "DAYS_EMPLOYED", "PAYMENT_RATE", "PREV_CNT_PAYMENT_MEAN", "ANNUITY_INCOME_PERC"]
        #st_ms = st.multiselect("Choose columns", main_model_df.columns.tolist(), default = cols)

        st.subheader("Select a page below:")
        mode = st.radio(
            "Menu",
            [   "Welcome Here: Overview",
                "Client Credit Info",
                "Client Loan Decision"
            ],
        )


    if mode == "Client Loan Decision":
        st.title("Client Loan Decision")
        #st.write()

        st.sidebar.subheader("Select a state or all US states:")

        # API information
        server_url = 'http://127.0.0.1:8000'
        endpoint = '/credit_decision/'

        # Creating parameters
        params = {
            'Credit ID':st.sidebar.slider('Credit ID', df['SK_ID_CURR'].min(), df['SK_ID_CURR'].max(), step = 1)
        }

        user_input = st.sidebar.text_input("Client Credit ID", '100002')   

        # defining process for calling API 
        #def process(data, server_url: str):
         #   result = requests.post(server_url, data)   
          #  return result  
        
        st.subheader("Client Prediction :  ")
        st.markdown("These are the probabilities and final prediction for the requested client based on our models : ")

        if st.sidebar.button("Predict"):   
            #result = process(df, server_url+endpoint+user_input)  
            result = requests.get(server_url+endpoint+user_input)  
            #result.status_code
            #result.text
            #result.reason  
            #result.content

            list_result = result.text.split('","')
            prob0 = list_result[0].split('":"')
            prob1 = list_result[1].split('":"')
            prob2 = list_result[2].split('":"')

            prob0_txt = prob0[0].strip('{"')
            prob0_flt = float(prob0[1])

            prob1_txt = prob1[0]
            prob1_flt = float(prob1[1])
            
            prob2_txt = prob2[0]
            prob2_flt = float(prob2[1].strip('}"'))

            st.write(prob0_txt + " : ", prob0_flt)
            st.write(prob1_txt + " : ", prob1_flt)
            st.write(prob2_txt + " : ", prob2_flt)
            #

        else:  
            pass 

    elif mode == "Client Credit Info":
        st.title(" Title Text Client Credit Info")

    elif mode == "Welcome Here: Overview":
        st.title(" Title Text")
  
        df = load_data()

        st.header("Header Text ")
        st.markdown("Markdown Text ")

        st.subheader("Subheader Text ")
        st.markdown("Markdown Text ")

        st.dataframe(df.iloc[:,2:].head(10))

        df.iloc[:,2:].shape

        st.markdown(
            """
            MArkdown Text

            """
        )
    
        st.latex(" Latex Text")

        st.write(
            "write text"
        )




#streamlit run share_new.py / dash_ver1.py
