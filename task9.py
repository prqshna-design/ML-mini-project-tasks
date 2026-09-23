
# TASK 9: TAXI FARE PREDICTION
# Linear Regression


# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 2. Create the dataset
data = {
    "Distance": [
        1.2, 2.5, 3.1, 4.0, 5.2,
        6.5, 7.0, 8.2, 9.5, 10.0,
        11.5, 12.0, 13.5, 15.0, 16.2,
        17.5, 18.0, 20.0, 22.0, 25.0,
        3.5, 4.8, 6.0, 7.5, 9.0,
        10.5, 12.5, 14.0, 16.5, 19.0
    ],

    "Trip_Duration": [
        8, 12, 15, 18, 22,
        25, 28, 30, 35, 38,
        40, 43, 47, 50, 55,
        58, 60, 65, 70, 75,
        16, 20, 24, 29, 34,
        40, 45, 52, 58, 68
    ],

    "Passengers": [
        1, 1, 2, 1, 2,
        3, 1, 2, 1, 2,
        3, 1, 2, 3, 1,
        2, 4, 1, 2, 3,
        1, 2, 1, 3, 2,
        1, 4, 2, 3, 1
    ],

    "Time_of_Day": [
        8, 9, 10, 11, 12,
        13, 14, 15, 16, 17,
        18, 19, 20, 21, 22,
        23, 7, 6, 5, 4,
        9, 11, 13, 15, 17,
        19, 21, 23, 8, 18
    ],

    "Fare": [
        8.5, 11.0, 13.0, 15.5, 19.0,
        22.5, 24.0, 27.5, 31.0, 33.0,
        36.5, 38.0, 42.0, 46.0, 49.0,
        53.0, 55.0, 60.0, 66.0, 75.0,
        14.0, 18.0, 21.0, 25.5, 30.0,
        34.0, 40.0, 44.0, 51.0, 62.0
    ]
}

df = pd.DataFrame(data)

# 3. Display the original dataset

print("===== ORIGINAL DATASET =====")
print(df)

# 4. Check for missing values

print("\n===== MISSING VALUES =====")

print(df.isnull().sum())



# 5. Clean the dataset


# Remove rows with missing values
df = df.dropna()
print("\nDataset after cleaning:")
print(df)


# ==========================================
# 6. Handle outliers
# ==========================================

# We will use the IQR method.
# IQR = Q3 - Q1
#
# Values below Q1 - 1.5*IQR
# or above Q3 + 1.5*IQR
# are considered outliers.

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR

print("\n===== OUTLIER LIMITS =====")

print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)


# Remove fare outliers
df_clean = df[
    (df["Fare"] >= lower_limit) &
    (df["Fare"] <= upper_limit)
]


print("\nDataset after removing outliers:")

print(df_clean)


# ==========================================
# 7. Explore the data
# ==========================================

print("\n===== STATISTICAL SUMMARY =====")

print(df_clean.describe())


# ==========================================
# 8. Visualize Distance vs Fare
# ==========================================

plt.scatter(
    df_clean["Distance"],
    df_clean["Fare"]
)

plt.xlabel("Distance")
plt.ylabel("Taxi Fare")
plt.title("Distance vs Taxi Fare")

plt.show()


# ==========================================
# 9. Visualize Trip Duration vs Fare
# ==========================================

plt.scatter(
    df_clean["Trip_Duration"],
    df_clean["Fare"]
)

plt.xlabel("Trip Duration (minutes)")
plt.ylabel("Taxi Fare")
plt.title("Trip Duration vs Taxi Fare")

plt.show()


# ==========================================
# 10. Select features and target
# ==========================================

X = df_clean[
    [
        "Distance",
        "Trip_Duration",
        "Passengers",
        "Time_of_Day"
    ]
]

y = df_clean["Fare"]


# ==========================================
# 11. Split the dataset
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n===== DATA SPLIT =====")

print("Training samples:", len(X_train))

print("Testing samples:", len(X_test))


# ==========================================
# 12. Build Linear Regression model
# ==========================================

model = LinearRegression()


# Train the model
model.fit(X_train, y_train)


# ==========================================
# 13. Predict taxi fares
# ==========================================

y_pred = model.predict(X_test)


print("\n===== ACTUAL VS PREDICTED FARES =====")

for actual, predicted in zip(y_test, y_pred):

    print(
        "Actual Fare:",
        round(actual, 2),
        " | Predicted Fare:",
        round(predicted, 2)
    )


# ==========================================
# 14. Evaluate the model
# ==========================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print("\n===== MODEL EVALUATION =====")

print("MAE:", round(mae, 2))

print("RMSE:", round(rmse, 2))

print("R² Score:", round(r2, 2))


# ==========================================
# 15. Analyze residuals
# ==========================================

# Residual = Actual - Predicted

residuals = y_test - y_pred


print("\n===== RESIDUALS =====")

for actual, predicted, residual in zip(
    y_test,
    y_pred,
    residuals
):

    print(
        "Actual:",
        round(actual, 2),
        "Predicted:",
        round(predicted, 2),
        "Residual:",
        round(residual, 2)
    )


# ==========================================
# 16. Plot residuals
# ==========================================

plt.scatter(
    y_pred,
    residuals
)

plt.axhline(
    y=0
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual Analysis")

plt.show()


# ==========================================
# 17. Compare Actual vs Predicted Fare
# ==========================================

plt.scatter(
    y_test,
    y_pred
)

plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("Actual vs Predicted Taxi Fare")


# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()


# ==========================================
# 18. Analyze model coefficients
# ==========================================

print("\n===== MODEL COEFFICIENTS =====")

for feature, coefficient in zip(
    X.columns,
    model.coef_
):

    print(
        feature,
        ":",
        round(coefficient, 4)
    )


print("\nIntercept:")

print(round(model.intercept_, 4))