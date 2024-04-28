import pandas as pd

# Read data from personalloan.xlsx
df = pd.read_excel('personalloan.xlsx')

# Drop 'PersonalLoan' and 'CreditCard' columns
df.drop(['PersonalLoan', 'CreditCard'], axis=1, inplace=True)

# Add 'Term deposit' column based on condition
df['Term deposit'] = ((df['CDAccount'] == 0) & (df['CCAvg in thousands'] < df['Income in Thousaunds'])).astype(int)

# Write data to Termdeposit.xlsx
df.to_excel('Termdeposit.xlsx', index=False)

print("Termdeposit.xlsx file has been created successfully.")
