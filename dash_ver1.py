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

# Loading of the data : read the df obtained from merging the csv files and keeping the rows where TARGET is a 
# missing value we keep all values and data as they are originally, since our model is already done and will not 
# be trained here
@st.cache
def load_data_small_sample():
    data_initial = pd.read_csv('../my_csv_files/df_small_sample_db.csv')
    #df_app = data_initial_plt.append(data_initial_plt2)
    # #return df_app
    return data_initial

data_small_smple = load_data_small_sample()

@st.cache
def load_data_train():
    data_initial = pd.read_csv('../my_csv_files/train_csv_db.csv')
    #data_initial_plt2 = pd.read_csv('../Projet+Mise+en+prod+-+home-credit-default-risk/application_test.csv')
    #df_app = data_initial_plt.append(data_initial_plt2)
    # #return df_app
    return data_initial

data_initial_plt = load_data_train()

if __name__ == "__main__":
    
    #st.info("Prêt à Dépenser Dashboard")
    st.title("Prêt à Dépenser Dashboard")

    with st.sidebar:
        st.title("Prêt à Dépenser Dashboard Pages")
        # Sidebar creation ####################################################################
        #st.sidebar.title('Prêt à Dépenser Dashboard Pages')
        st.sidebar.markdown("Please Select One of the Following Pages: ")

        #cols = ["EXT_SOURCE_2", "DAYS_EMPLOYED", "PAYMENT_RATE", "PREV_CNT_PAYMENT_MEAN", "ANNUITY_INCOME_PERC"]
        #st_ms = st.multiselect("Choose columns", main_model_df.columns.tolist(), default = cols)

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
        - Welcome to the Prêt à Dépenser dashboard.
        - See below for some overall stats of our clients, and to explore the client data we have available. 
        """
        )

        #st.header("Header Text ")
        #st.markdown("Markdown Text ")

        df_modl = data_small_smple.copy()

        st.subheader("Overall Stats : ")
        st.write("Present Number of Loans : ", df_modl.iloc[:,2:].shape[0])
        st.write("Present Number of Features Available to Explore : ", df_modl.iloc[:,2:].shape[1])

        #st.subheader("Subheader Text ")
        #st.markdown(" Example of Customer Information  : ")
        #st.latex(" Latex Text")

        #st.write(
        #    "write text"
        #)

        data_initial = data_initial_plt.copy()

        fig = px.pie(data_initial, names = 'NAME_CONTRACT_TYPE', title='Contract Types (in Percentage of Clients)')
        st.plotly_chart(fig, use_container_width=True)

        fig = px.pie(data_initial, names = 'NAME_INCOME_TYPE', title='Clients Income Type (in Percentage of Clients)')
        st.plotly_chart(fig, use_container_width=True)


        st.subheader("Exploration of our Client Data & Information : ")
        df = data_small_smple.copy().drop(['Unnamed: 0', 'index'],  axis = 1)

        feature = st.selectbox(
            "Choose a feature : ", list(df.columns))
            #['SK_ID_CURR', 'TARGET', 'CODE_GENDER']

        fig = px.histogram(
                data_frame=df,
                x = feature,
                title="Histogram of the selected feature : ",
            )
            #plot = plotly_plot(chart_type, df)
        st.plotly_chart(fig, use_container_width=True)
        
        if not feature:
            st.error("Please select one feature.")
        else:
            #data = df.loc[features]
            #st.write(type(feature))

            fig = px.line(
                data_frame = df,
                x= 'SK_ID_CURR', y = feature,
                title = "Selected feature vs the client's credit ID : ",
            )
            #plot = plotly_plot(chart_type, df)
            st.plotly_chart(fig, use_container_width=True)

            st.write("Selected features with associated loan IDs, sorted by the first selected feature, in descending order : ")

            #st.dataframe(df[["SK_ID_CURR", str(feature)]].sort_values(by = feature, ascending = False).reset_index(drop = True))
            # .style.hide_index()

            features2 = st.multiselect(
            "Choose a feature : ", list(df.columns), ["SK_ID_CURR", 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
            'DAYS_EMPLOYED_PERC', 'DAYS_EMPLOYED',
            'AMT_ANNUITY', 'AMT_CREDIT',
            'INSTAL_DPD_MEAN'])
            #st.write((features2))
            st.dataframe(df.loc[:, features2].sort_values(by = feature, ascending = False).reset_index(drop = True))
        
        #df["DAYS_BIRTH_YEAR"] = -(df["DAYS_BIRTH"] / 365)
        #
        #fig = px.histogram(
        #        data_frame=df,
        #        x="DAYS_BIRTH_YEAR",
        #        title="Client's age in years at the time of application : ",
        #    )
        ##plot = plotly_plot(chart_type, df)
        #st.plotly_chart(fig, use_container_width=True)
        


    elif mode == "Client Credit Info":

        #st.write(prob0_txt + " : ", prob0_flt)

        st.subheader("Please Type in the Credit ID and press Enter : ")

        # API information
        server_url = 'http://127.0.0.1:8000'
        #server_url = 'https://mighty-tundra-05371.herokuapp.com'
        endpoint = '/credit_decision/'

        df_mode2 = data_small_smple.copy()
        min_v = int(df_mode2['SK_ID_CURR'].min())
        max_v = int(df_mode2['SK_ID_CURR'].max())

        # Creating parameters
        params = {
            'Credit ID':st.slider('Credit ID', min_value = min_v, max_value = max_v, step = 1)
        }
        # .sidebar

        #user_input = st.sidebar.selectbox(
            #"Choose which feature shall be used to order the data : ", features_for_s_client)   

        user_input = st.text_input("Client Credit ID", '400001')   

        #st.title(" Title Text Client Credit Info")

        #if st.button("Show"):   
        #result = process(df, server_url+endpoint+user_input)  
        result = requests.get(server_url+endpoint+user_input)  
        #result.status_code
        #result.text
        #result.reason
        #imp_ft_client = st.selectbox(
        #"Choose which feature shall be used to order the data : ", features_for_s_client)
        #['SK_ID_CURR', 'TARGET', 'CODE_GENDER']

        features_for_s_client = st.multiselect("Choose the features you wish to see : ", list(df_mode2.columns), 
        ["SK_ID_CURR", 'EXT_SOURCE_2', 'EXT_SOURCE_3', 
            'DAYS_EMPLOYED_PERC', 'DAYS_EMPLOYED',
            'AMT_ANNUITY', 'AMT_CREDIT',
            'INSTAL_DPD_MEAN'])
        if not features_for_s_client :
            st.error("Please select one feature.")
        else:
            index_needed = df_mode2.index[df_mode2["SK_ID_CURR"] == int(user_input)].values[0]
            st.dataframe(df_mode2.loc[(index_needed-5):(index_needed+5), features_for_s_client].reset_index(drop = True))

            feature_line = st.selectbox(
            "Choose a specific feature : ", list(df_mode2.columns))
            #['SK_ID_CURR', 'TARGET', 'CODE_GENDER']

            fig = px.line(
                data_frame = df_mode2,
                x= 'SK_ID_CURR', y = feature_line,
                title = "Selected feature vs the client's credit ID : ",
            )

            val_hline = list(df_mode2[df_mode2["SK_ID_CURR"] == int(user_input)][feature_line])
            #st.write(val_hline)
            #st.write(val_hline[0])
            fig.add_hline(y = val_hline[0])

            #plot = plotly_plot(chart_type, df)
            st.plotly_chart(fig, use_container_width=True)
        #data = df.loc[features]
        #st.write(type(feature))
         

        #index_point = df_mode2[df_mode2["SK_ID_CURR"] == int(user_input)].reset_index()
        #st.write(index_point)
        #index_needed = index_point["level_0"]
        #a[start:stop:step]
        
        # .sort_values(by = imp_ft_client, ascending = False


    elif mode == "Client Loan Decision":
        
        st.subheader("Client Loan Decision : ")

        #st.title("Client Loan Decision")
        #st.write()

        st.sidebar.subheader("Please Type in the Credit ID : ")

        # API information
        server_url = 'http://127.0.0.1:8000'
        #server_url = 'https://mighty-tundra-05371.herokuapp.com'
        endpoint = '/credit_decision/'

        df_mode3 = data_small_smple.copy()

        min_v = int(df_mode3['SK_ID_CURR'].min())
        max_v = int(df_mode3['SK_ID_CURR'].max())

        # Creating parameters
        params = {
            'Credit ID':st.sidebar.slider('Credit ID', min_value = min_v, max_value = max_v, step = 1)
        }

        user_input = st.sidebar.text_input("Client Credit ID", '400001')   

        # defining process for calling API 
        #def process(data, server_url: str):
         #   result = requests.post(server_url, data)   
          #  return result  
        
        st.markdown("These are the probabilities and final prediction for the requested client based on our models : ")

        #train_df = pd.read_csv('../my_csv_files/MY_train_x.csv')
        #df_selec_plus = pd.read_csv('../my_csv_files/df_selec_plus_sk_id.csv')
        clf = joblib.load('../Models/lgbm_trained_myscore_final.pickle')

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
            #st.write(index_point)
            index_p = index_point["level_0"]
            #st.write(index_p)

            shap_data_plot  = df_selec_plus.copy().drop(['SK_ID_CURR', 'TARGET', 'index', 'Unnamed: 0'],  axis = 1)

            explainer = shap.TreeExplainer(clf)

            shap_values = explainer.shap_values(shap_data_plot)

            st.subheader("SHAP Plot : ")

            def st_shap(plot, height = None):
                    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
                    components.html(shap_html, height = height)

            st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][index_p,:], shap_data_plot.iloc[index_p,:], link = "logit"))
            

            # get the most important features for this sample
            vals = np.abs(pd.DataFrame(shap_values[1][index_p,:]).values).mean(0)
            feature_names = shap_data_plot.columns
            feature_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                              columns=['col_name','feature_importance_vals'])
            feature_importance.sort_values(by=['feature_importance_vals'],
                                           ascending=False, inplace=True)

            
            st.subheader("Feature importance for the current client : ")

            st.dataframe(feature_importance.head(10))

            most_important_features = list(feature_importance["col_name"].head(5))

            st.subheader("Graphs of important features for the current client : ")

            df = shap_data_plot
            # Here we use a column with categorical data
            #fig = px.histogram(df, x = "AMT_CREDIT")
            fig = px.histogram(df, x = most_important_features[0], nbins = 40)
            val_hline = list(df_selec_plus[df_selec_plus["SK_ID_CURR"] == int(user_input)][most_important_features[0]])
            #st.write(val_hline)
            #st.write(val_hline[0])
            fig.add_vline(x = val_hline[0])
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(df, x = most_important_features[1], nbins = 40)
            val_hline = list(df_selec_plus[df_selec_plus["SK_ID_CURR"] == int(user_input)][most_important_features[1]])
            fig.add_vline(x = val_hline[0])
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(df, x = most_important_features[2], nbins = 40)
            val_hline = list(df_selec_plus[df_selec_plus["SK_ID_CURR"] == int(user_input)][most_important_features[2]])
            fig.add_vline(x = val_hline[0])
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(df, x = most_important_features[3], nbins = 40)
            val_hline = list(df_selec_plus[df_selec_plus["SK_ID_CURR"] == int(user_input)][most_important_features[3]])
            fig.add_vline(x = val_hline[0])
            st.plotly_chart(fig, use_container_width=True)

            fig = px.histogram(df, x = most_important_features[4], nbins = 40)
            val_hline = list(df_selec_plus[df_selec_plus["SK_ID_CURR"] == int(user_input)][most_important_features[4]])
            fig.add_vline(x = val_hline[0])
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("SHAP Summary Plot of the Prediction Model Used : ")
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot(shap.summary_plot(shap_values[1], shap_data_plot))
          
   
            #st_shap(shap.force_plot(explainer.expected_value[1], shap_values[1][950,:], shap_data_plot.iloc[950,:]))

            # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
            #st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], shap_data.iloc[0,:]))
            #shap.force_plot(explainer.expected_value[1], shap_values[1][950,:], shap_data.iloc[950,:])
                #

        else:  
            pass 

        
       

    


#streamlit run share_new.py / dash_ver1.py
