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

import shap
import joblib

st.set_page_config(
    page_title="Title",
    layout="wide",
    initial_sidebar_state="expanded",
)

# read the df obtained from merging the csv files and keeping the rows where TARGET is a missing value
# we keep all values and data as they are originally, since our model is already done and will not be trained here

# Loading of the data
@st.cache
def load_data1():

    data = pd.read_csv('../my_csv_files/df_for_modelling.csv')

    return data

df1 = load_data1()

@st.cache
def load_data2():
    data_initial = pd.read_csv('../Projet+Mise+en+prod+-+home-credit-default-risk/application_train.csv')
    #data_initial_plt2 = pd.read_csv('../Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    #df_app = data_initial_plt.append(data_initial_plt2)
    # #return df_app
    return data_initial

data_initial_plt = load_data2()


if __name__ == "__main__":
    
    #st.info("Prêt à Dépenser Dashboard")
    st.title("Prêt à Dépenser Dashboard")

    with st.sidebar:
        st.title("Prêt à Dépenser Dashboard Pages")
        # Sidebar creation ####################################################################
        #st.sidebar.title('Prêt à Dépenser Dashboard Pages')
        st.sidebar.markdown("Please Select One of the Following : ")

        #cols = ["EXT_SOURCE_2", "DAYS_EMPLOYED", "PAYMENT_RATE", "PREV_CNT_PAYMENT_MEAN", "ANNUITY_INCOME_PERC"]
        #st_ms = st.multiselect("Choose columns", main_model_df.columns.tolist(), default = cols)

        #st.subheader("Select a page below:")
        mode = st.radio(
            "",
            [   "Overview of Prêt à Dépenser: Clients and Loans",
                "Client Credit Info",
                "Client Loan Decision"
            ],
        )

    
    if mode == "Overview of Prêt à Dépenser: Clients and Loans":
        
        st.write(
        """
        - Welcome to the Prêt à Dépenser Dashboard.
        - See below for some overall stats of our clients. 
        """
        )

        df = df1.copy()

        #st.header("Header Text ")
        #st.markdown("Markdown Text ")

        st.write("Present Number of Loans : ", df.iloc[:,2:].shape[0])

        #st.subheader("Subheader Text ")
        st.markdown(" Example of Customer Information  : ")

        st.dataframe(df.iloc[:,2:].head(10))

        #st.write("Present Number of Loans ", df.iloc[:,2:].shape[0])
        

        #st.markdown(
        #    """
        #    MArkdown Text
#
        #    """
        #)
    
        #st.latex(" Latex Text")

        #st.write(
        #    "write text"
        #)

        #with st.echo():
        #    fig = px.histogram(
        #        data_frame=df,
        #        x="AMT_INCOME_TOTAL",
        #        title="Count of Bill Depth Observations",
        #    )
        
        data_initial = data_initial_plt.copy()

        fig = px.pie(data_initial, names = 'NAME_CONTRACT_TYPE', title='Contract Types (in Percentage of Clients)')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.pie(data_initial, names = 'NAME_INCOME_TYPE', title='Clients Income Type (in Percentage of Clients)')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.histogram(
                data_frame=df,
                x="AMT_INCOME_TOTAL",
                title="Income of clients requesting loans : ",
            )
        #plot = plotly_plot(chart_type, df)
        st.plotly_chart(fig, use_container_width=True)
        
        df["DAYS_BIRTH_YEAR"] = -(df["DAYS_BIRTH"] / 365)
        
        fig = px.histogram(
                data_frame=df,
                x="DAYS_BIRTH_YEAR",
                title="Client's age in years at the time of application : ",
            )
        #plot = plotly_plot(chart_type, df)
        st.plotly_chart(fig, use_container_width=True)
        


    elif mode == "Client Credit Info":
        
        df = df1.copy()

        st.title(" Title Text Client Credit Info")

        #st.write(prob0_txt + " : ", prob0_flt)

        st.sidebar.subheader("Please Type in the Credit ID :")

        # Creating parameters
        params = {
            'Credit ID':st.sidebar.slider('Credit ID', df['SK_ID_CURR'].min(), df['SK_ID_CURR'].max(), step = 1)
        }

        user_input = st.sidebar.text_input("Client Credit ID", '100002')   


    elif mode == "Client Loan Decision":
        st.title("Client Loan Decision")
        #st.write()

        st.sidebar.subheader("Please Type in the Credit ID :")

        # API information
        server_url = 'http://127.0.0.1:8000'
        endpoint = '/credit_decision/'

        df = df1.copy()

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

        train_df = pd.read_csv('../my_csv_files/MY_train_x.csv')
        clf = joblib.load('../Models/lgbm_trained_my_score03.pickle')

        explainer = shap.TreeExplainer(clf)

        #shap_values = explainer.shap_values(train_df)

        #shap.force_plot(explainer.expected_value[1], shap_values[1][950,:], train_df.iloc[950,:])

        #shap.force_plot(expected_value, shap_values[idx,:], features = X.iloc[idx,4:], 
        #link='logit', matplotlib=True, figsize=(12,3))
        #st.pyplot(bbox_inches='tight',dpi=300,pad_inches=0)
        #plt.clf()

    

   


#streamlit run share_new.py / dash_ver1.py
