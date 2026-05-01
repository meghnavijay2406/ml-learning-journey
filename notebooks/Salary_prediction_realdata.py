import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


df = pd.read_csv("data/Salary Data.csv")
df = df.dropna()

print("Data sample:")
print(df.head())
df.info()
df.describe()


X = df[["Age", "Gender", "Education Level", "Job Title", "Years of Experience"]]
y = df["Salary"]

categorical_features = ["Gender", "Education Level", "Job Title"]
numeric_features = ["Age", "Years of Experience"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)



models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42)
}

print("\nModel Comparison (Cross-Validation):")

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", model)
    ])
    
    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="r2"
    )
    
    print(f"\n{name}")
    print("R2 scores:", scores)
    print("Mean R2:", np.mean(scores))
    
    best_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

final_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", best_model)
])

# Train/test split (final evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

final_model.fit(X_train, y_train)

y_pred = final_model.predict(X_test)

print("\nSample Predictions:")
print(y_pred[:5])

print("\nActual Values:")
print(y_test.values[:5])

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nFinal Model Performance:")
print("MAE:", mae)
print("R2 Score:", r2)

import os
import joblib

base_dir = os.path.dirname(os.path.dirname(__file__))  # project root
model_path = os.path.join(base_dir, "models", "salary_pipeline.pkl")

os.makedirs(os.path.dirname(model_path), exist_ok=True)

joblib.dump(final_model, model_path)

print("Saved at:", model_path)