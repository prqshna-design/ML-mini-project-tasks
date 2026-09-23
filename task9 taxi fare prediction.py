import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 1. Create manual dataset
data = {
    "trip_duration": [
        10, 15, 20, 25, 30,
        35, 40, 45, 50, 55,
        60, 65, 70, 75, 80,
        85, 90, 95, 100, 110,
        120, 130, 140, 150, 160,
        170, 180, 190, 200, 220
    ],

    "distance_traveled": [
        1.0, 1.5, 2.0, 2.5, 3.0,
        3.5, 4.0, 4.5, 5.0, 5.5,
        6.0, 6.5, 7.0, 7.5, 8.0,
        8.5, 9.0, 9.5, 10.0, 11.0,
        12.0, 13.0, 14.0, 15.0, 16.0,
        17.0, 18.0, 19.0, 20.0, 22.0
    ],

    "num_of_passengers": [
        1, 1, 2, 1, 2,
        3, 1, 2, 1, 2,
        3, 1, 2, 3, 1,
        2, 4, 1, 2, 3,
        1, 2, 1, 3, 2,
        1, 4, 2, 3, 1
    ],

    "fare": [
        8, 10, 12, 15, 17,
        20, 22, 25, 27, 30,
        32, 35, 37, 40, 43,
        46, 48, 51, 54, 58,
        62, 66, 70, 75, 80,
        85, 90, 95, 100, 110
    ]
}

df = pd.DataFrame(data)


# 2. Clean dataset
print("Missing values:")
print(df.isnull().sum())

df = df.dropna()


# 3. Handle outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outlier_mask = (
    (df < (Q1 - 0.4 * IQR)) |
    (df > (Q3 + 0.4 * IQR))
)

df = df[~outlier_mask.any(axis=1)]

print("\nDataset after removing outliers:")
print(df.describe())


# 4. Train-Test Split
X = df[["trip_duration", "distance_traveled", "num_of_passengers"]]
y = df["fare"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 5. Build Linear Regression Model
model = linear_model.LinearRegression()
model.fit(X_train, y_train)


# 6. Predict
y_pred = model.predict(X_test)


# 7. Evaluate Model
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


# 8. Analyze Residuals
residuals = y_test - y_pred

plt.scatter(y_pred, residuals)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Analysis")
plt.show()


# 9. Compare Actual vs Predicted Fare
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Fare")
plt.show()