#region imports
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
#endregion


# block for generating model to personal loan predictor
# developed and tested by Nikhil shravan khobragade
#region personal loan predictor
data = pd.read_excel('personalloan.xlsx', sheet_name=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, 'personalloan.pkl')

#endregion

# block for generating model to Term deposit loan predictor
# developed and tested by Kavya n shetty
#region termdeposit predictor
data = pd.read_excel('Termdeposit.xlsx', sheet_name=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, 'Termdeposit.pkl')
#endregion

# block for generating model to creditcard loan predictor
# developed and tested by Pavan kumar pendyala

#region  credit card predictor

data = pd.read_excel('creditcard.xlsx', sheet_name=0)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

joblib.dump(model, 'creditcard.pkl')
#endregion
