« Implémentez un modèle de scoring »

# Overview 

This project is binary classification problem with two main objectives. The first, is to build a machine learning model that will predict the probability of credit payment failure for clients of a financial company, "Prêt à dépenser". The second objective is to build an interactive dashboard intended to improve the company's knowledge on its clients, and to help explain the decision of granted or refusing a credit loan to their clients.  

# Materials & Exploratory Data Analysis

This project was first carried out locally, using JupyterLab, Visual Studio Code, Python 3.9 and necessary packages (including: NumPy, Pandas, Matplotlib, Seaborn, PyCaret, Scikit-Learn, LightGBM, HyperOpt, FastApi). The dashboard was created using Streamlit. Once completed, the project was deployed to Heroku using Git. 

The data used for this project is the data made available by the “Home Credit Default Risk · Can you predict how capable each applicant is of repaying a loan?” competition (comprised of 8 csv files). 

The exploratory data analysis for this project was inspired by the following Kaggle kernel: “LightGBM with Simple Features” created by Aguiar, as part of the competition. 
7 out of the 8 available data files were merged into one dataframe. The last file, containing descriptions for the columns in the various other data files, was only used as reference. All categorical data was handled using OneHotEncoder. Some features were created and/ or aggregated used pre-existing features. 
The final merged dataframe contained 356251 rows and 798 columns. Missing values varied from none to 85% of the total values, depending on the features. 

A notable observation was that the two classes of the target variable were severely imbalanced: 282665 instances for target value = 0, vs 24823 instances for target value = 1, giving us a ration of 10 to 1.

