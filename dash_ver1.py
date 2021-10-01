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
def load_data_small_sample():
    data_initial = pd.read_csv('df_small_sample_db.csv')
    #data_initial_plt2 = pd.read_csv('../Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    #df_app = data_initial_plt.append(data_initial_plt2)
    # #return df_app
    return data_initial

data_small_smple = load_data_small_sample()

@st.cache
def load_data_train():
    data_initial = pd.read_csv('application_train.csv')
    #data_initial_plt2 = pd.read_csv('../Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    #df_app = data_initial_plt.append(data_initial_plt2)
    # #return df_app
    return data_initial

data_initial_plt = load_data_train()

@st.cache
def load_data_for_modelling():

    data = pd.read_csv('df_for_api.csv')
    return data

df_m = load_data_for_modelling()


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

        #st.header("Header Text ")
        #st.markdown("Markdown Text ")

        df_modl = df_m.copy()
        st.write("Present Number of Loans : ", df_modl.iloc[:,2:].shape[0])

        #st.subheader("Subheader Text ")
        st.markdown(" Example of Customer Information  : ")

        st.dataframe(df_modl.iloc[:,2:].head(10))

        #st.write("Present Number of Loans ", df.iloc[:,2:].shape[0])
        

        #st.markdown(
        #    """
        #    MArkdown Text
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

        df = data_small_smple.copy()
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
        
        df_mode2 = df_m.copy()

        #st.write(prob0_txt + " : ", prob0_flt)

        st.sidebar.subheader("Please Type in the Credit ID :")

        # API information
        #server_url = 'http://127.0.0.1:8000'
        server_url = 'https://mighty-tundra-05371.herokuapp.com'
        endpoint = '/credit_decision/'

        min_v = int(df_mode2['SK_ID_CURR'].min())
        max_v = int(df_mode2['SK_ID_CURR'].max())

        # Creating parameters
        params = {
            'Credit ID':st.sidebar.slider('Credit ID', min_value = min_v, max_value = max_v, step = 1)
        }

        user_input = st.sidebar.text_input("Client Credit ID", '100002')   

        st.title(" Title Text Client Credit Info")


    elif mode == "Client Loan Decision":
        st.title("Client Loan Decision")
        #st.write()

        st.sidebar.subheader("Please Type in the Credit ID :")

        # API information
        #server_url = 'http://127.0.0.1:8000'
        server_url = 'https://mighty-tundra-05371.herokuapp.com'
        endpoint = '/credit_decision/'

        df_mode3 = df_m.copy()

        min_v = int(df_mode3['SK_ID_CURR'].min())
        max_v = int(df_mode3['SK_ID_CURR'].max())

        # Creating parameters
        params = {
            'Credit ID':st.sidebar.slider('Credit ID', min_value = min_v, max_value = max_v, step = 1)
        }

        user_input = st.sidebar.text_input("Client Credit ID", '100002')   

        # defining process for calling API 
        #def process(data, server_url: str):
         #   result = requests.post(server_url, data)   
          #  return result  
        
        st.subheader("Client Prediction :  ")
        st.markdown("These are the probabilities and final prediction for the requested client based on our models : ")

        #train_df = pd.read_csv('../my_csv_files/MY_train_x.csv')
        #df_selec_plus = pd.read_csv('../my_csv_files/df_selec_plus_sk_id.csv')
        clf = joblib.load('lgbm_trained_myscore_final.pickle')

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

            df_selec_plus = df_mode3

            index_point = df_selec_plus[df_selec_plus["SK_ID_CURR"] == int(user_input)].reset_index()
            index_p = index_point["index"]
            st.write(index_p)

            shap_data_plot  = df_selec_plus.copy().drop(['SK_ID_CURR', 'TARGET', 'Unnamed: 0'],  axis = 1)

            explainer = shap.TreeExplainer(clf)

            shap_values = explainer.shap_values(shap_data_plot)

            def st_shap(plot, height = None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height = height)

            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][index_p,:], shap_data_plot.iloc[index_p,:]))
            #st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][950,:], shap_data_plot.iloc[950,:]))

            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            #st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], shap_data.iloc[0,:]))
            #shap.force_plot(explainer.expected_value[1], shap_values[1][950,:], shap_data.iloc[950,:])
                #

        else:  
            pass 

        
       

    


#streamlit run share_new.py / dash_ver1.py
