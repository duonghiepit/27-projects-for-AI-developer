import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Load dataset
housing = fetch_california_housing(as_frame=True)

# Create a Dataframe from the dataset
df = housing.frame

print("Califonia Housing Data:")
print(df.head())

# Features (Independant Variables) and target (dependent variable)
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model using mse and r2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}\nR2 Score: {r2}")

print("Model coefficients:")
print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")

coef_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficent'])

print("Coefficients for each feature")
print(coef_df)

# Test the model with new data
new_data = pd.DataFrame(
    {
        'MedInc': [5],
        'HouseAge': [30],
        'AveRooms': [6],
        'AveBedrms': [2],
        'Population': [500],
        'AveOccup': [3],
        'Latitude': [34.05],
        'Longitude': [-118.25]
    }
)

predicted_price = model.predict(new_data)
print(f"Predicted house price: ${predicted_price[0]:.2f}")