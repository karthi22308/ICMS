#region imports
import joblib
import openai
import pandas as pd
#endregion


# console application that uses the model and predicts the output
# developed and tested by Karthick C

#region for binary conversion
def get_binary_input(prompt):
    # Helper function to get binary input (1 or 0) from Y/N input
    while True:
        try:
            value = input(prompt).strip().upper()  # Convert input to uppercase and remove leading/trailing spaces
            if value == 'Y':
                return 1
            elif value == 'N':
                return 0
            else:
                print("Invalid input. Please enter 'Y' or 'N'.")
        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C)
            print("\nOperation canceled.")
            exit()
        except Exception as e:
            print("Error:", e)

#endregion

#region for handling binary exceptions
def get_input(prompt):
    # Helper function to get input from user and handle exceptions
    while True:
        try:
            value = input(prompt)
            return value
        except KeyboardInterrupt:
            # Handle keyboard interrupt (Ctrl+C)
            print("\nOperation canceled.")
            exit()
        except Exception as e:
            print("Error:", e)

#endregion

#region to generate text from open ai
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
# console control begins
# developed and tested by karthick C

#region to get inputs from console
print("Welcome to Intelligent Client Management System\n")

# Get inputs from the user
age = get_input("Enter age: ")
experience = get_input("Enter years of experience with bank: ")
income = str(get_input("Enter income in thousands: "))
zip_code = get_input("Enter ZIP Code: ")
family_members = get_input("Enter number of family members: ")
cc_avg = get_input("Enter Monthly average expenditure in thousands: ")
education = get_input("Enter education level (1 for higher secondary, 2 for UG, 3 for PG): ")
mortgage = get_input("Enter mortgage amount: ")
securities_account = get_binary_input("Do you have a SecuritiesAccount? (Y/N): ")
cd_account = get_binary_input("Do you have a CDAccount? (Y/N): ")
online = get_binary_input("Do you use online banking? (Y/N): ")
credit_card = get_binary_input("Do you have a credit card? (Y/N): ")
personal_loan = get_binary_input("Do you have a personal loan? (Y/N): ")
#endregion

#region to convert inputs to dataframe and predict output
personalinput = [age, experience, income, zip_code, family_members, cc_avg, education, mortgage, securities_account,
                 cd_account, online, credit_card]

ccinput = [age, experience, income, zip_code, family_members, cc_avg, education, mortgage, securities_account,
           cd_account, online]
input_datapl = pd.DataFrame([personalinput], columns=['Age', 'Experience', 'Income in Thousaunds', 'ZIP Code', 'Family',
                                                      'CCAvg in thousands', 'Education', 'Mortgage',
                                                      'SecuritiesAccount', 'CDAccount', 'Online(opted Netbanking)',
                                                      'CreditCard'])
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
text = 'i will give you some details about a person, please give some banking service suggestions we can provide him age-' + str(
    age) + ', years related with bank-' + str(experience) + ', Income in Thousaunds-' + str(
    income) + ', ZIP code-' + str(zip) + ',family members-' + str(
    family_members) + ',credit card usage in average-' + str(
    cc_avg) + ',education- ' + education + ',mortage-' + str(mortgage)
output = ' '
if personal_loan == 0 and predictionp == 1: output += 'personal loan can be suggested '
if credit_card == 0 and predictionc == 1: output += 'credit card can be suggested'
if predictiont == 1: output += ' Term Deposit can be suggested' + str(predictionc)

if output == ' ':
    print("unable to predict services with certainity")
else:
    print(output)
bot_response = generate_response(text)
print("services that could be offered:")
print(bot_response)
#endregion
