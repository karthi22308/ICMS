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

import matplotlib.pyplot as plt
import seaborn as sns
import mplcursors
st.set_option('deprecation.showPyplotGlobalUse', False)
modeln=''
def drawgraph(uploaded_file,model):
    st.write(modeln)
    #if modeln=='RandomForest':
    st.write("Feature Importances:")
    data = pd.read_excel(uploaded_file)
    st.write("Bar Chart:")
    plot_feature_importances_bar(model, data.iloc[:, :-1])
    st.write("Pie Chart:")
    plot_feature_importances_pie(model, data.iloc[:, :-1])



def plot_feature_importances_pie(model, feature_names):
    if isinstance(feature_names, pd.DataFrame):
        feature_names = feature_names.columns

    feature_importances = model.feature_importances_
    sorted_idx = feature_importances.argsort()

    # Plotting a pie chart instead of a horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(feature_importances[sorted_idx], labels=[feature_names[i] for i in sorted_idx],
                                      autopct='%1.1f%%', startangle=90)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title('Feature Importances')

    # Enable cursor support for the pie chart
    mplcursors.cursor(hover=True).connect("add", lambda sel: sel.annotation.set_text(
        f"{sel.artist.get_labels()[sel.target.index]}: {sel.artist.get_autopct()(feature_importances[sorted_idx][sel.target.index], feature_importances[sorted_idx])}"))

    st.pyplot(fig)

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
        modeln = modelname
    data = pd.read_excel(uploaded_file, sheet_name=0)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_pred)
    error =mean_squared_error(y_test, y_pred)
    if (accuracy > maxaccuracy):
        rmodel = model
        maxaccuracy = accuracy
        modelname = 'Linear Regression'
        modeln = modelname
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
        st.info(modeln)
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





