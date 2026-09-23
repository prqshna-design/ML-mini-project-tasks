import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. Load dataset
url = "https://raw.githubusercontent.com/Renukumar-R/Taxi-Fare-Prediction-Using-Linear-Regression/main/taxi_fare_data.csv"
df = pd.read_csv(url)

print(df.info())
print(df.describe())


# 2. Remove outliers
IQR = df.quantile(0.75) - df.quantile(0.25)
threshold = 0.4

outlier_mask = (
    (df < (df.quantile(0.25) - threshold * IQR)) |
    (df > (df.quantile(0.75) + threshold * IQR))
)

df = df[~outlier_mask.any(axis=1)]


# 3. Train-Test Split
X = df[["trip_duration", "distance_traveled", "num_of_passengers"]]
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2023
)


# 4. Train Linear Regression
model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# 5. Predict and evaluate
y_pred = model.predict(X_test)

mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


# 6. Residual Analysis
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Plot")
plt.show()


# 7. Actual vs Predicted
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.show()


# 8. Predict a new taxi fare
new_trip = pd.DataFrame({
    "trip_duration": [50],
    "distance_traveled": [17],
    "num_of_passengers": [2]
})

print("\nPredicted Fare:", model.predict(new_trip)[0])