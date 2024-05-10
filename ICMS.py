# region headers


import pandas as pd
import streamlit as st
import openai
import joblib
import os

# endregion
# region Global declarations
folder_path = r'C:\Users\1000070232\PycharmProjects\Hackathon\output'


# endregion
# region helper functions
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


def find_total_customers(uploaded_file):
    if uploaded_file is not None:
        # Read the uploaded Excel file into a pandas DataFrame
        df = pd.read_excel(uploaded_file)
        # Get the number of rows in the DataFrame, which corresponds to the number of customers
        total_customers = len(df)
        return total_customers


# endregion

# region Page Design
# region Mode selection
st.title("Intelligent Client Management System")
mde = st.radio(
    "Select Mode of application",
    ('Form', 'Export'))
# endregion

# region Form mode
# developed and tested by Nikhil shravan khobragade
if mde == "Form":
    # region inputs
    st.title("Please fill details of the customer")
    age = st.slider('Age of the Customer', 0, 130, 25)
    experience = st.slider('Customer Account Age (in years):', 0, 130, 5)
    income = st.slider('income of the Customer(in thousands)', 0, 500, 25)
    zip = st.number_input('zip code')
    Family = st.slider('Total Family members of the customer', 1, 10, 2)
    avgcc = st.slider('average Monthly expenditure of customer(in thousands)', 0, 500, 25)
    education = st.radio(
        "Highest education level of customer",
        ('Higher secondary', 'under graduate', 'post graduate'))
    mortgage = st.slider('mortage(in thousands)', 0, 500, 25)
    security = st.toggle('security account')
    cdaccount = st.toggle('Term deposit account')
    onlinebanking = st.toggle('online banking')
    creditcard = st.toggle('credit card')
    personalloan = st.toggle('personal loan')
    text = 'i will give you some details about a person, please give some banking service suggestions we can provide him age-' + str(
        age) + ', years related with bank-' + str(experience) + ', Income in Thousaunds-' + str(
        income) + ', ZIP code-' + str(zip) + ',family members-' + str(Family) + ',credit card usage in average-' + str(
        avgcc) + ',education- ' + education + ',mortage-' + str(mortgage)
    # endregion
    # region data preparation and passing
    if education == "Higher secondary":
        edu = 1
    else:
        if education == "under graduate":
            edu = 2
        else:
            edu = 3
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
    if (creditcard):
        cc = 1
    else:
        cc = 0
    # cc=int(creditcard == 'True')
    if (personalloan):
        pl = 1
    else:
        pl = 0
    # pl=int(personalloan == 'True')
    personalinput = [age, experience, income, zip, Family, avgcc, edu, mortgage, sec, cd, net, cc]

    ccinput = [age, experience, income, zip, Family, avgcc, edu, mortgage, sec, cd, net]
    input_datapl = pd.DataFrame([personalinput],
                                columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
                                         'CCAvg in thousands', 'Education', 'Mortgage', 'SecuritiesAccount',
                                         'CDAccount', 'Online(opted Netbanking)', 'CreditCard'])
    input_datacc = pd.DataFrame([ccinput],
                                columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
                                         'CCAvg in thousands', 'Education', 'Mortgage', 'SecuritiesAccount',
                                         'CDAccount', 'Online(opted Netbanking)'])
    modelp = joblib.load('personalloan.pkl')
    modelc = joblib.load('creditcard.pkl')
    modelt = joblib.load('Termdeposit.pkl')
    predictionp = modelp.predict(input_datapl)[0]
    predictionc = modelc.predict(input_datacc)[0]
    predictiont = modelt.predict(input_datacc)[0]

    output = 'Services that can be suggested are:'
    var = 1
    # endregion
    # region prediction
    if st.button('Predict'):
        if pl == 0 and predictionp == 1:
            output += '\n' + str(var) + '. Personal loan can be suggested '
            var += 1
        if cc == 0 and predictionc == 1:
            output += '\n' + str(var) + '. Credit card can be suggested'
            var += 1
        if predictiont == 1:
            output += '\n' + str(var) + '. Term Deposit can be suggested'

        if output == 'Services that can be suggested are:':
            st.info("unable to predict services with certainity", icon="ℹ️")
        else:
            st.info(output, icon="ℹ️")
        bot_response = generate_response(text)
        st.write("Services that could be offered:", bot_response)

    # endregion
# endregion

# region bulk mode
# developed and tested by Karthick C

else:

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:

        if st.button('Predict'):

            with st.spinner("ICMS Running..."):
                # region Loading UI message
                total_customers = find_total_customers(uploaded_file)
                timet = int(total_customers * 1.8)
                st.info(
                    f"Please note that this Application Uses OpenAI its response time can vary  depending on various factors, including the current load on OpenAI's servers, network latency, and the specific capabilities of the API expected time for the uploaded file is {timet} seconds..")

                # new_data = pd.read_csv(uploaded_file)
                new_data = pd.read_excel(uploaded_file)
#endregion
                # region Load ML models
                # Extract customerid and phone number columns
                customer_info = new_data[['CustomerId', 'Mobile number']]

                # Load machine learning models
                model_loan = joblib.load('personalloan.pkl')
                model_credit_card = joblib.load('creditcard.pkl')
                model_term_deposit = joblib.load('Termdeposit.pkl')
                # endregion
                #region input preparation and prediction from ML

                # Remove unnecessary columns from new_data for model_loan prediction
                new_data_loan = new_data.drop(['CustomerId', 'Mobile number'], axis=1)
                predictions_loan = model_loan.predict(new_data_loan)
                predictions_df_loan = pd.DataFrame(predictions_loan, columns=['Personal loan acceptance'])

                # Remove unnecessary columns from new_data for model_credit_card prediction
                new_data_credit_card = new_data.iloc[:,
                                       2:-1]  # Excluding customerid, mobile_number, and the last column
                predictions_credit_card = model_credit_card.predict(new_data_credit_card)
                # Adjust predictions to 0 if 'CreditCard' is 1
                predictions_credit_card[new_data['CreditCard'] == 1] = 0
                predictions_df_credit_card = pd.DataFrame(predictions_credit_card, columns=['Credit card acceptance'])

                # Remove unnecessary columns from new_data for model_term_deposit prediction
                new_data_term_deposit = new_data.iloc[:, 2:-1]  # Excluding the last column
                predictions_term_deposit = model_term_deposit.predict(new_data_term_deposit)
                predictions_df_term_deposit = pd.DataFrame(predictions_term_deposit,
                                                           columns=['Term deposit acceptance'])
                #endregion
                #region prepare output excels

                # Create output directory if it doesn't exist
                output_dir = 'output'
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Create separate Excel files for each prediction
                result_df_loan = pd.concat([customer_info, predictions_df_loan], axis=1)
                result_df_loan[predictions_df_loan['Personal loan acceptance'] == 1].to_excel(
                    os.path.join(output_dir, 'personal_loan_accepted.xlsx'), index=False)
                # result_df_loan[predictions_df_loan['Personal loan acceptance'] == 0].to_excel(
                # os.path.join(output_dir, 'personal_loan_rejected.xlsx'), index=False)

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
                rejected_customer_ids = set(
                    result_df_loan[result_df_loan['Personal loan acceptance'] == 0]['CustomerId']) & \
                                        set(result_df_credit_card[result_df_credit_card['Credit card acceptance'] == 0][
                                                'CustomerId']) & \
                                        set(result_df_term_deposit[
                                                result_df_term_deposit['Term deposit acceptance'] == 0][
                                                'CustomerId'])

                # Filter original input data for rejected customers
                rejected_customers_df = new_data[new_data['CustomerId'].isin(rejected_customer_ids)]

                # Save rejected customers details to fourth Excel file
                rejected_customers_df.to_excel(os.path.join(output_dir, 'rejected_customers.xlsx'), index=False)

                # Filter original input data for rejected customers
                rejected_customers_df = new_data[new_data['CustomerId'].isin(rejected_customer_ids)]

                rejected_customers_df = new_data[new_data['CustomerId'].isin(rejected_customer_ids)]
                #endregion
                # region open ai region for rejected cust
                # Generate text for each rejected customer
                rejected_customers_df[
                    'text'] = "I will give you some details about a person, please give some banking service suggestions we can provide him(our Ml model predicted that we cant provide credit card, personal loan or term deposit suggestion to the customer so exclude that):\n"
                rejected_customers_df['text'] += ("Age: " + rejected_customers_df['Age'].astype(str) + "\n")
                rejected_customers_df['text'] += (
                        "Years related with bank: " + rejected_customers_df['Experience'].astype(str) + "\n")
                rejected_customers_df['text'] += (
                        "Income in Thousaunds: " + rejected_customers_df['Income in Thousaunds'].astype(str) + "\n")
                rejected_customers_df['text'] += ("ZIP code: " + rejected_customers_df['ZIP Code'].astype(str) + "\n")
                rejected_customers_df['text'] += (
                            "Family members: " + rejected_customers_df['Family'].astype(str) + "\n")
                rejected_customers_df['text'] += (
                        "Credit card usage in average: " + rejected_customers_df['CCAvg in thousands'].astype(
                    str) + "\n")
                rejected_customers_df['text'] += ("Education: " + rejected_customers_df['Education'].astype(str) + "\n")
                rejected_customers_df['text'] += ("Mortgage: " + rejected_customers_df['Mortgage'].astype(str) + "\n")

                # Pass text to generate_response method
                rejected_customers_df['Gen_AI_Response'] = rejected_customers_df['text'].apply(generate_response)
                # Save rejected customers details to fourth Excel file without 'text' column
                rejected_customers_df.drop(columns=['text']).to_excel(
                    os.path.join(output_dir, 'rejected_customers.xlsx'),
                    index=False)
                # endregion


                # region open ai region for personal loan
                # Read personal_loan_accepted.xlsx to get CustomerId values

                personal_loan_df = pd.read_excel('output/personal_loan_accepted.xlsx')
                customer_ids = personal_loan_df['CustomerId']

                # Read upload_file.xlsx to get customer details
                upload_file_df = pd.read_excel(uploaded_file)

                # Iterate through CustomerId values
                for customer_id in customer_ids:
                    # Find the row corresponding to the CustomerId
                    customer_row = upload_file_df[upload_file_df['CustomerId'] == customer_id]

                    if not customer_row.empty:
                        # Construct the sentence using customer details
                        sentence = f"I will give you some details about a person, we have generated a Ml model from historic data of those got personal loan this model predicted tht this person might opt for personal loan so give me information about the person and suggest some more services that a bank can offer and help to understand about the customer\n"
                        sentence += f"Age: {customer_row['Age'].iloc[0]}\n"
                        sentence += f"Years related with bank: {customer_row['Experience'].iloc[0]}\n"
                        sentence += f"income of person in thousands {customer_row['Income in Thousaunds'].iloc[0]}\n"

                        sentence += f"total family members {customer_row['Family'].iloc[0]}\n"
                        sentence += f"average monthly expenditure in thousands {customer_row['CCAvg in thousands'].iloc[0]}\n"
                        sentence += f"education level 1 for higher secondary 2 for graduation 3 for post graduation : {customer_row['Education'].iloc[0]}\n"
                        sentence += f"mortage amount : {customer_row['Mortgage'].iloc[0]}\n"
                        sentence += f"has security account 1/0: {customer_row['SecuritiesAccount'].iloc[0]}\n"
                        sentence += f"has CDAccount 1/0: {customer_row['CDAccount'].iloc[0]}\n"
                        sentence += f"has Online(opted Netbanking) 1/0: {customer_row['Online(opted Netbanking)'].iloc[0]}\n"
                        sentence += f"has CreditCard 1/0: {customer_row['CreditCard'].iloc[0]}\n"
                        # Pass the sentence to generate_response method
                        response = generate_response(sentence)

                        # Append the generated response to personal_loan_accepted.xlsx
                        personal_loan_df.loc[
                            personal_loan_df['CustomerId'] == customer_id, 'Gen_AI_Response'] = response

                # Save the updated personal_loan_accepted.xlsx file
                personal_loan_df.to_excel('output/personal_loan_accepted.xlsx', index=False)
                # endregion
                # region open ai region for credit card customers
                # Read personal_loan_accepted.xlsx to get CustomerId values

                personal_loan_df = pd.read_excel('output/credit_card_accepted.xlsx')
                customer_ids = personal_loan_df['CustomerId']

                # Read upload_file.xlsx to get customer details
                upload_file_df = pd.read_excel(uploaded_file)

                # Iterate through CustomerId values
                for customer_id in customer_ids:
                    # Find the row corresponding to the CustomerId
                    customer_row = upload_file_df[upload_file_df['CustomerId'] == customer_id]

                    if not customer_row.empty:
                        # Construct the sentence using customer details
                        sentence = f"I will give you some details about a person, we have generated a Ml model from historic data of those got credit card this model predicted tht this person might opt for credit card so give me information about the person and suggest some more services that a bank can offer and help to understand about the customer\n"
                        sentence += f"Age: {customer_row['Age'].iloc[0]}\n"
                        sentence += f"Years related with bank: {customer_row['Experience'].iloc[0]}\n"
                        sentence += f"income of person in thousands {customer_row['Income in Thousaunds'].iloc[0]}\n"

                        sentence += f"total family members {customer_row['Family'].iloc[0]}\n"
                        sentence += f"average monthly expenditure in thousands {customer_row['CCAvg in thousands'].iloc[0]}\n"
                        sentence += f"education level 1 for higher secondary 2 for graduation 3 for post graduation : {customer_row['Education'].iloc[0]}\n"
                        sentence += f"mortage amount : {customer_row['Mortgage'].iloc[0]}\n"
                        sentence += f"has security account 1/0: {customer_row['SecuritiesAccount'].iloc[0]}\n"
                        sentence += f"has CDAccount 1/0: {customer_row['CDAccount'].iloc[0]}\n"
                        sentence += f"has Online(opted Netbanking) 1/0: {customer_row['Online(opted Netbanking)'].iloc[0]}\n"
                        sentence += f"has CreditCard 1/0: {customer_row['CreditCard'].iloc[0]}\n"
                        # Pass the sentence to generate_response method
                        response = generate_response(sentence)

                        # Append the generated response to personal_loan_accepted.xlsx
                        personal_loan_df.loc[
                            personal_loan_df['CustomerId'] == customer_id, 'Gen_AI_Response'] = response

                # Save the updated personal_loan_accepted.xlsx file
                personal_loan_df.to_excel('output/credit_card_accepted.xlsx', index=False)
                # endregion

                # # region open ai region for term deposit customers
                # # Read personal_loan_accepted.xlsx to get CustomerId values
                #
                # personal_loan_df = pd.read_excel('output/term_deposit_accepted.xlsx')
                # customer_ids = personal_loan_df['CustomerId']
                #
                # # Read upload_file.xlsx to get customer details
                # upload_file_df = pd.read_excel(uploaded_file)
                #
                # # Iterate through CustomerId values
                # for customer_id in customer_ids:
                #     # Find the row corresponding to the CustomerId
                #     customer_row = upload_file_df[upload_file_df['CustomerId'] == customer_id]
                #
                #     if not customer_row.empty:
                #         # Construct the sentence using customer details
                #         sentence = f"I will give you some details about a person, we have generated a Ml model from historic data of those got term deposit this model predicted tht this person might opt for term deposit so give me information about the person and suggest some more services that a bank can offer and help to understand about the customer\n"
                #         sentence += f"Age: {customer_row['Age'].iloc[0]}\n"
                #         sentence += f"Years related with bank: {customer_row['Experience'].iloc[0]}\n"
                #         sentence += f"income of person in thousands {customer_row['Income in Thousaunds'].iloc[0]}\n"
                #
                #         sentence += f"total family members {customer_row['Family'].iloc[0]}\n"
                #         sentence += f"average monthly expenditure in thousands {customer_row['CCAvg in thousands'].iloc[0]}\n"
                #         sentence += f"education level 1 for higher secondary 2 for graduation 3 for post graduation : {customer_row['Education'].iloc[0]}\n"
                #         sentence += f"mortage amount : {customer_row['Mortgage'].iloc[0]}\n"
                #         sentence += f"has security account 1/0: {customer_row['SecuritiesAccount'].iloc[0]}\n"
                #         sentence += f"has CDAccount 1/0: {customer_row['CDAccount'].iloc[0]}\n"
                #         sentence += f"has Online(opted Netbanking) 1/0: {customer_row['Online(opted Netbanking)'].iloc[0]}\n"
                #         sentence += f"has CreditCard 1/0: {customer_row['CreditCard'].iloc[0]}\n"
                #         # Pass the sentence to generate_response method
                #         response = generate_response(sentence)
                #
                #         # Append the generated response to personal_loan_accepted.xlsx
                #         personal_loan_df.loc[personal_loan_df['CustomerId'] == customer_id, 'Gen_AI_Response'] = response
                #
                # # Save the updated personal_loan_accepted.xlsx file
                # personal_loan_df.to_excel('output/term_deposit_accepted.xlsx', index=False)
                # # endregion
                #region for displaying generated excels

                st.info("Completed", icon="ℹ️")
                st.info("Excel files have created in output folder separately ", icon="ℹ️")
                # st.write("Predictions from trained Model:")
                # data1 = pd.read_excel(folder_path + r"\credit_card_accepted.xlsx", usecols=[0, 1])
                #
                # # Read the first two columns from the second Excel file
                # data2 = pd.read_excel(folder_path + r"\personal_loan_accepted.xlsx", usecols=[0, 1])
                #
                # # Read the first two columns from the third Excel file
                # data3 = pd.read_excel(folder_path + r"\term_deposit_accepted.xlsx", usecols=[0, 1])
                #
                # # Concatenate the first two columns into a single DataFrame and give names to each column
                # combined_data = pd.concat(
                #     [data1.iloc[:, 0], data1.iloc[:, 1], data2.iloc[:, 0], data2.iloc[:, 1], data3.iloc[:, 0],
                #      data3.iloc[:, 1]], axis=1)
                # combined_data.columns = ["Credit card - customerid", "Credit card - mobile number", "Personal loan - customerid", "Personal loan - mobile number",
                #                          "Term Deposit - customerid", "Term Deposit - mobile number"]  # Specify column names here
                # combined_data.columns = combined_data.columns.str.replace(',', '')
                # combined_data = combined_data.applymap(lambda x: str(x).replace(',', '') if isinstance(x, str) else x)
                #
                # st.dataframe(combined_data)
                st.info("customers  suggested for personal loan")
                file_path = folder_path + r"\personal_loan_accepted.xlsx"
                data = pd.read_excel(file_path)
                first_column = data.iloc[:, 0]
                last_column = data.iloc[:, -1]
                secon_column = data.iloc[:, 1]

                # Concatenate the selected columns into a new DataFrame
                selected_data = pd.concat([first_column, secon_column, last_column], axis=1)
                st.dataframe(selected_data)

                st.info("customers  suggested for Credit card")
                file_path = folder_path + r"\credit_card_accepted.xlsx"
                data = pd.read_excel(file_path)
                first_column = data.iloc[:, 0]
                last_column = data.iloc[:, -1]
                secon_column = data.iloc[:, 1]

                # Concatenate the selected columns into a new DataFrame
                selected_data = pd.concat([first_column, secon_column, last_column], axis=1)
                st.dataframe(selected_data)
                st.info("customers  suggested for Term Deposit")
                file_path = folder_path + r"\term_deposit_accepted.xlsx"
                data = pd.read_excel(file_path)
                first_column = data.iloc[:, 0]
                last_column = data.iloc[:, -1]

                # Concatenate the selected columns into a new DataFrame
                selected_data = pd.concat([first_column, last_column], axis=1)
                st.dataframe(selected_data)

                file_path = folder_path + r"\rejected_customers.xlsx"

                data = pd.read_excel(file_path)
                # Select the first and last column
                first_column = data.iloc[:, 0]

                last_column = data.iloc[:, -1]

                # Concatenate the selected columns into a new DataFrame
                selected_data = pd.concat([first_column, last_column], axis=1)
                # Display the contents of the Excel file
                st.write("customers cant suggested for personal loan/credit card/Termdeposit")

                st.dataframe(selected_data)
                #endregion

# endregion

# endregion
