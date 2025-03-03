import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("housing.csv")

# Handle missing values
df["total_bedrooms"].fillna(df["total_bedrooms"].median(), inplace=True)

# One-hot encoding for categorical feature
df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# Define features and target variable
X = df.drop(columns=["median_house_value"])
y = df["median_house_value"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Standardization
scaler = StandardScaler()

# Ridge Regression Model
ridge = Ridge()

# Create a pipeline
pipeline = Pipeline([
    ('poly_features', poly),
    ('scaler', scaler),
    ('ridge', ridge)
])

# Define hyperparameter grid for Ridge Regression
param_grid = {'ridge__alpha': [0.1, 1, 5, 10, 50, 100]}

# Grid Search Cross Validation
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Predictions
y_pred = best_model.predict(X_test)

# Evaluation Metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Best Alpha: {grid_search.best_params_['ridge__alpha']}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
