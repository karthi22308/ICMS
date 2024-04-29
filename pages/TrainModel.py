import ipaddress
import os

import joblib
import openai
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import mplcursors
import numpy as np

st.set_option('deprecation.showPyplotGlobalUse', False)
modeln=''
def drawgraph(uploaded_file,model):
    #region preparing data
    data = pd.read_excel(uploaded_file, sheet_name=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    #endregion
    st.write("Feature Importances:")
    data = pd.read_excel(uploaded_file)
    st.write("Pie Chart:")
    plot_feature_importances_pie(model, data.iloc[:, :-1])
    st.write("Bar Chart:")
    plot_feature_importances_bar(model, data.iloc[:, :-1])

def plot_feature_importances_pie(model, feature_names):
    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    # Create a DataFrame for Plotly Express
    data = {
        "Feature": [feature_names[i] for i in sorted_idx],
        "Importance": feature_importances[sorted_idx]
    }
    df = pd.DataFrame(data)

    # Create a pie chart using Plotly Express
    fig = px.pie(df, values='Importance', names='Feature', title='Feature Importances')

    # Display the plot
    st.plotly_chart(fig)

def plot_feature_importances_bar(model, feature_names):
    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    plt.figure(figsize=(10, 8))
    plt.barh(range(len(sorted_idx)), feature_importances[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importances')
    st.pyplot()

def getmodel(uploaded_file):
    maxaccuracy = 0
    data = pd.read_excel(uploaded_file, sheet_name=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if(accuracy>maxaccuracy):
        rmodel=model
        maxaccuracy=accuracy
        modelname='RandomForest'
        modeln = modelname
    data = pd.read_excel(uploaded_file, sheet_name=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if (accuracy > maxaccuracy):
        rmodel = model
        maxaccuracy = accuracy
        modelname = 'Gaussian Naive Bayes'
    # Read data from the uploaded file
    data = pd.read_excel(uploaded_file, sheet_name=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    # Make predictions
    y_pred = knn.predict(X_test)
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > maxaccuracy:
        rmodel = knn
        maxaccuracy = accuracy
        modelname = 'k-Nearest Neighbors'

    st.info('model trained with Accuracy: ' + str(maxaccuracy*100)+'% using '+modelname+' Algorithm')
    return  rmodel




st.title("ML Model trainer for data driven recommendation engine")
mde = st.radio(
    "Select service that needs a model to be trained",
    ('Personal loan', 'credit card','Term Deposit'))
if mde == "Personal loan":
    uploaded_file = st.file_uploader("Choose a xlsx file with required data")
    st.text('please Make sure that the data is in the format:')
    st.info('Age,Experience,Income in Thousaunds,ZIP Code,Family,CCAvg in thousands,Education,Mortgage,SecuritiesAccount,CDAccount,Online(opted Netbanking),CreditCard,PersonalLoan accepted Y/N')
    if uploaded_file is not None and st.button('Train'):
        model=getmodel(uploaded_file)
        joblib.dump(model, 'personalloan.pkl')
        #file_path = "your_dataset.xlsx"
        drawgraph(uploaded_file,model)





if mde == "credit card":
    uploaded_file = st.file_uploader("Choose a xlsx file with required data")
    st.text('please Make sure that the data is in the format:')
    st.info(
        'Age,Experience,Income in Thousaunds,ZIP Code,Family,CCAvg in thousands,Education,Mortgage,SecuritiesAccount,CDAccount,Online(opted Netbanking),CreditCard accepted Y/N')
    if uploaded_file is not None and st.button('Train'):
        model=getmodel(uploaded_file)
        joblib.dump(model, 'creditcard.pkl')
        drawgraph(uploaded_file, model)



if mde == "Term Deposit":
    uploaded_file = st.file_uploader("Choose a xlsx file with required data")
    st.text('please Make sure that the data is in the format:')
    st.info(
        'Age,Experience,Income in Thousaunds,ZIP Code,Family,CCAvg in thousands,Education,Mortgage,SecuritiesAccount,CDAccount,Online(opted Netbanking),Term deposit accepted Y/N')
    if uploaded_file is not None and st.button('Train'):
        model=getmodel(uploaded_file)
        joblib.dump(model, 'Termdeposit.pkl')
       # file_path = "your_dataset.xlsx"
        drawgraph(uploaded_file, model)





