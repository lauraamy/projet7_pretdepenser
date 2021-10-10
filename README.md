« Implémentez un modèle de scoring »

# Overview 

This project is binary classification problem with two main objectives. The first, is to build a machine learning model that will predict the probability of credit payment failure for clients of a financial company, "Prêt à dépenser". The second objective is to build an interactive dashboard intended to improve the company's knowledge on its clients, and to help explain the decision of granted or refusing a credit loan to their clients.  

# Materials 

This project was first carried out locally, using JupyterLab, Visual Studio Code, Python 3.9 and necessary packages (including: NumPy, Pandas, Matplotlib, Seaborn, PyCaret, Scikit-Learn, LightGBM, HyperOpt, FastApi). The dashboard was created using Streamlit. Once completed, the project was deployed to Heroku using Git. 

The data used for this project is the data made available by the “Home Credit Default Risk · Can you predict how capable each applicant is of repaying a loan?” competition (comprised of 8 csv files). 

The exploratory data analysis for this project was inspired by the following Kaggle kernel: “LightGBM with Simple Features” created by Aguiar, as part of the competition. 

# Files

Exploration and modelling notebooks : 
p7_notebook_exploration.ipynb
p7_notebook_modelling.ipynb
p7_notebook_pycaret.ipynb

Model for API : 
lgbmhyperpar_mythresh_0_5.pickle

Code for APi : main_copy.py

Files to run the dashboard : 
df_small_sample_db.csv
train_csv_db.csv

