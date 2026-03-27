import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("--- Electric Guitar Price Predictor ---")

# 1. Creating a synthetic dataset of electric guitars

data = {
    'Brand': ['Fender', 'Gibson', 'Ibanez', 'PRS', 'Epiphone', 'Squier', 'Jackson', 'ESP', 'Fender', 'Gibson', 'Ibanez', 'PRS'],
    'Pickup_Type': ['Single Coil', 'Humbucker', 'Humbucker', 'Humbucker', 'Humbucker', 'Single Coil', 'Active', 'Active', 'Humbucker', 'P90', 'Single Coil', 'Humbucker'],
    'Condition_Out_of_10': [8, 7, 9, 10, 6, 9, 8, 9, 7, 8, 6, 9],
    'Age_Years': [10, 15, 2, 1, 5, 3, 4, 2, 8, 12, 7, 3],
    'Price_USD': [800, 1500, 600, 2200, 400, 250, 700, 1000, 850, 1600, 450, 2000]
}
df = pd.DataFrame(data)
print("\nDataset loaded successfully. Here are the first few rows:")
print(df.head())

# 2. Data Preprocessing (Converting text data like 'Brand' into numbers for the AI)
df_encoded = pd.get_dummies(df, columns=['Brand', 'Pickup_Type'])

# 3. Splitting the data into Features (X) and Target Price (y)
X = df_encoded.drop('Price_USD', axis=1)
y = df_encoded['Price_USD']

# Splitting into training data (80%) and testing data (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Training the Machine Learning Model
print("\nTraining the Random Forest ML Model...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Predictions and Evaluating the Model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)

print("\n--- Results ---")
print(f"Model Mean Absolute Error: ${mae:.2f}")
print("\nSample Predictions on Test Data:")
for actual, predicted in zip(y_test, predictions):
    print(f"Actual Market Price: ${actual} | AI Predicted Price: ${predicted:.2f}")