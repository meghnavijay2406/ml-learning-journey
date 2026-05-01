import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "YearsExperience": [1, 2, 3, 4, 5, 6, 7, 8 ,9, 10],
    "Salary": [10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000]
}

df = pd.DataFrame(data)
df

plt.scatter(df["YearsExperience"], df["Salary"])
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Experience VS Salary")
plt.show()

X = df[["YearsExperience"]]
y = df[["Salary"]]

X_train, X_test, y_train, y_test =   train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual:", y_test.values)

from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, y_pred))

import joblib
joblib.dump(model, "salary_model.pkl")

model.predict(pd.DataFrame({"YearsExperience": [5]}))
