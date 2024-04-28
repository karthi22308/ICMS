#region headers
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import openai
import joblib
import os
#endregion
#region azure implementation
def generate_response(input_text):

    openai.api_type = "azure"
    openai.api_base = "https://testingkey.openai.azure.com/"
    openai.api_version = "2023-09-15-preview"
    openai.api_key = "35f96eb3ffc04868981139be478f99fa"

    response = openai.Completion.create(
        engine="TestingChatModel",
        prompt=input_text,
        temperature=0.2,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        best_of=1,
        stop=None)
    return response['choices'][0]['text']

#endregion

#region Page Design
st.title("Intelligent Client Management System")
mde = st.radio(
    "Select Mode of application",
    ('Form', 'Export'))
#region Form mode
if mde == "Form":
    #region inputs
    st.title("Please fill details of the customer")
    age = st.slider('Age of the Customer', 0, 130, 25)
    experience = st.slider('Customer Account Age (in years):', 0, 130, 5)
    income = st.slider('income of the Customer(in thousands)', 0, 500, 25)
    zip = st.number_input('zip code')
    Family = st.slider('Total Family members of the customer', 1, 10, 2)
    avgcc = st.slider('average Monthly expenditure of customer(in thousands)', 0, 500, 25)
    education = st.radio(
        "Highest education level of customer",
        ('Higher secondary', 'under graduate','post graduate'))
    mortgage = st.slider('mortage(in thousands)', 0, 500, 25)
    security = st.toggle('security account')
    cdaccount = st.toggle('Term deposit account')
    onlinebanking = st.toggle('online banking')
    creditcard = st.toggle('credit card')
    personalloan = st.toggle('personal loan')
    text='i will give you some details about a person, please give some banking service suggestions we can provide him age-'+str(age)+', years related with bank-'+str(experience)+', Income in Thousaunds-'+str(income)+', ZIP code-'+str(zip)+',family members-'+str(Family)+',credit card usage in average-'+str(avgcc)+',education- '+education+',mortage-'+str(mortgage)
    #endregion
    #region data preparation and passing
    if education=="Higher secondary": edu=1
    else :
        if education=="under graduate": edu=2
        else:
            edu=3
   # sec = int(security == 'True')
    if (security):
        sec = 1
    else:
        sec = 0
   # cd=int(cdaccount == 'True')
    if (cdaccount):
        cd = 1
    else:
        cd = 0
   # net = int(onlinebanking == 'True')
    if (onlinebanking):
        net = 1
    else:
        net = 0
    if(creditcard):
        cc=1
    else :
        cc=0
   # cc=int(creditcard == 'True')
    if (personalloan):
        pl = 1
    else:
        pl = 0
   # pl=int(personalloan == 'True')
    personalinput = [age, experience, income,zip,Family,avgcc,edu,mortgage,sec,cd,net,cc]

    ccinput = [age, experience, income,zip,Family,avgcc,edu,mortgage,sec,cd,net]
    input_datapl = pd.DataFrame([personalinput], columns=['Age', 'Experience', 'Income in Thousaunds','ZIP Code','Family','CCAvg in thousands','Education','Mortgage','SecuritiesAccount','CDAccount','Online(opted Netbanking)','CreditCard'])
    input_datacc= pd.DataFrame([ccinput],
                                columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
                                         'CCAvg in thousands', 'Education', 'Mortgage', 'SecuritiesAccount',
                                         'CDAccount', 'Online(opted Netbanking)'])
    modelp = joblib.load('personalloan.pkl')
    modelc = joblib.load('creditcard.pkl')
    modelt = joblib.load('Termdeposit.pkl')
    predictionp = modelp.predict(input_datapl)[0]
    predictionc = modelc.predict(input_datacc)[0]
    predictiont = modelt.predict(input_datacc)[0]

    output=' '
    #endregion
    #region prediction
    if st.button('Predict'):
        if pl == 0 and predictionp==1: output+='personal loan can be suggested '
        if cc == 0 and  predictionc==1 :output+='credit card can be suggested'
        if predictiont == 1: output += ' Term Deposit can be suggested'+str(predictionc)

        if output==' ':
            st.info("unable to predict services with certainity", icon="ℹ️")
        else:
            st.info(output, icon="ℹ️")
        bot_response = generate_response(text)
        st.write("services that could be offered:", bot_response)

    #endregion
#endregion


#region bulk mode
else:
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        if st.button('Predict'):
           # new_data = pd.read_csv(uploaded_file)
           new_data = pd.read_excel(uploaded_file)

           # Extract customerid and phone number columns
           customer_info = new_data[['CustomerId', 'Mobile number']]

           # Load machine learning models
           model_loan = joblib.load('personalloan.pkl')
           model_credit_card = joblib.load('creditcard.pkl')
           model_term_deposit = joblib.load('Termdeposit.pkl')

           # Remove unnecessary columns from new_data for model_loan prediction
           new_data_loan = new_data.drop(['CustomerId', 'Mobile number'], axis=1)
           predictions_loan = model_loan.predict(new_data_loan)
           predictions_df_loan = pd.DataFrame(predictions_loan, columns=['Personal loan acceptance'])

           # Remove unnecessary columns from new_data for model_credit_card prediction
           new_data_credit_card = new_data.iloc[:, 2:-1]  # Excluding customerid, mobile_number, and the last column
           predictions_credit_card = model_credit_card.predict(new_data_credit_card)
           # Adjust predictions to 0 if 'CreditCard' is 1
           predictions_credit_card[new_data['CreditCard'] == 1] = 0
           predictions_df_credit_card = pd.DataFrame(predictions_credit_card, columns=['Credit card acceptance'])

           # Remove unnecessary columns from new_data for model_term_deposit prediction
           new_data_term_deposit = new_data.iloc[:, 2:-1]  # Excluding the last column
           predictions_term_deposit = model_term_deposit.predict(new_data_term_deposit)
           predictions_df_term_deposit = pd.DataFrame(predictions_term_deposit, columns=['Term deposit acceptance'])

           # Create output directory if it doesn't exist
           output_dir = 'output'
           if not os.path.exists(output_dir):
               os.makedirs(output_dir)

           # Create separate Excel files for each prediction
           result_df_loan = pd.concat([customer_info, predictions_df_loan], axis=1)
           result_df_loan[predictions_df_loan['Personal loan acceptance'] == 1].to_excel(
               os.path.join(output_dir, 'personal_loan_accepted.xlsx'), index=False)
          # result_df_loan[predictions_df_loan['Personal loan acceptance'] == 0].to_excel(
               #os.path.join(output_dir, 'personal_loan_rejected.xlsx'), index=False)

           result_df_credit_card = pd.concat([customer_info, predictions_df_credit_card], axis=1)
           result_df_credit_card[predictions_df_credit_card['Credit card acceptance'] == 1].to_excel(
               os.path.join(output_dir, 'credit_card_accepted.xlsx'), index=False)
          # result_df_credit_card[predictions_df_credit_card['Credit card acceptance'] == 0].to_excel(
             #  os.path.join(output_dir, 'credit_card_rejected.xlsx'), index=False)

           result_df_term_deposit = pd.concat([customer_info, predictions_df_term_deposit], axis=1)
           result_df_term_deposit[predictions_df_term_deposit['Term deposit acceptance'] == 1].to_excel(
               os.path.join(output_dir, 'term_deposit_accepted.xlsx'), index=False)
          # result_df_term_deposit[predictions_df_term_deposit['Term deposit acceptance'] == 0].to_excel(
              # os.path.join(output_dir, 'term_deposit_rejected.xlsx'), index=False)

           # Identify customer IDs with predictions as 0 for all products
           rejected_customer_ids = set(result_df_loan[result_df_loan['Personal loan acceptance'] == 0]['CustomerId']) & \
                                   set(result_df_credit_card[result_df_credit_card['Credit card acceptance'] == 0][
                                           'CustomerId']) & \
                                   set(result_df_term_deposit[result_df_term_deposit['Term deposit acceptance'] == 0][
                                           'CustomerId'])

           # Filter original input data for rejected customers
           rejected_customers_df = new_data[new_data['CustomerId'].isin(rejected_customer_ids)]

           # Save rejected customers details to fourth Excel file
           rejected_customers_df.to_excel(os.path.join(output_dir, 'rejected_customers.xlsx'), index=False)

           # Filter original input data for rejected customers
           rejected_customers_df = new_data[new_data['CustomerId'].isin(rejected_customer_ids)]

           rejected_customers_df = new_data[new_data['CustomerId'].isin(rejected_customer_ids)]

           # Generate text for each rejected customer
           rejected_customers_df[
               'text'] = "I will give you some details about a person, please give some banking service suggestions we can provide him(our Ml model predicted that we cant provide credit card, personal loan or term deposit suggestion to the customer so exclude that):\n"
           rejected_customers_df['text'] += ("Age: " + rejected_customers_df['Age'].astype(str) + "\n")
           rejected_customers_df['text'] += (
                       "Years related with bank: " + rejected_customers_df['Experience'].astype(str) + "\n")
           rejected_customers_df['text'] += (
                       "Income in Thousaunds: " + rejected_customers_df['Income in Thousaunds'].astype(str) + "\n")
           rejected_customers_df['text'] += ("ZIP code: " + rejected_customers_df['ZIP Code'].astype(str) + "\n")
           rejected_customers_df['text'] += ("Family members: " + rejected_customers_df['Family'].astype(str) + "\n")
           rejected_customers_df['text'] += (
                       "Credit card usage in average: " + rejected_customers_df['CCAvg in thousands'].astype(
                   str) + "\n")
           rejected_customers_df['text'] += ("Education: " + rejected_customers_df['Education'].astype(str) + "\n")
           rejected_customers_df['text'] += ("Mortgage: " + rejected_customers_df['Mortgage'].astype(str) + "\n")

           # Pass text to generate_response method
           rejected_customers_df['bot_response'] = rejected_customers_df['text'].apply(generate_response)

           # Save rejected customers details to fourth Excel file without 'text' column
           rejected_customers_df.drop(columns=['text']).to_excel(os.path.join(output_dir, 'rejected_customers.xlsx'),
                                                                 index=False)
           st.info("done", icon="ℹ️")


#endregion

#endregion





