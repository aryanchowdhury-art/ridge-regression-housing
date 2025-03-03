import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("housing.csv")

# Handle missing values
df.loc[:, "total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())

# Feature Engineering: Creating new meaningful features
df["rooms_per_household"] = df["total_rooms"] / df["households"]
df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
df["population_per_household"] = df["population"] / df["households"]

# **Fix: Remove log transformation on 'median_income'**
df["total_rooms"] = np.log1p(df["total_rooms"])
df["total_bedrooms"] = np.log1p(df["total_bedrooms"])
df["population"] = np.log1p(df["population"])
df["households"] = np.log1p(df["households"])

# Convert categorical variable into dummy variables
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Define target variable and features
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for Ridge Regression
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# **Fix: Reduce Ridge alpha from 10 to 1**
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Make predictions
y_pred = ridge_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Print results
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
