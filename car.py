# Car Price Prediction using Machine Learning

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# Load Dataset
df = pd.read_csv("car data.csv")

# Display first 5 rows
print("Dataset Preview:")
print(df.head())

# Dataset Information
print("\nDataset Info:")
print(df.info())

# Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Feature Engineering
# Convert Year to Car Age
current_year = 2026
df['Car_Age'] = current_year - df['Year']

# Drop unnecessary columns
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)

# Convert categorical columns into numerical
df = pd.get_dummies(df, drop_first=True)

print("\nProcessed Dataset:")
print(df.head())

# Define Features and Target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("------------------")
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("R2 Score:", r2)

# Feature Importance
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)

# Plot Feature Importance
plt.figure(figsize=(10,6))
importance.plot(kind='bar')
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# Actual vs Predicted Plot
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.tight_layout()
plt.show()

# Example Prediction
sample_car = X.iloc[[0]]
predicted_price = model.predict(sample_car)

print("\nSample Predicted Price:")
print(predicted_price[0])