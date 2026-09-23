# ==========================================
# HOUSE PRICE PREDICTION
# Linear Regression
# ==========================================

# 1. Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import numpy as np


# ==========================================
# 2. Create the dataset
# ==========================================

data = {
    "House_Area": [850, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
                   1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700],

    "Bedrooms": [2, 2, 2, 3, 3, 3, 3, 3, 4, 4,
                 4, 4, 4, 4, 5, 5, 5, 5, 5, 6],

    "Bathrooms": [1, 1, 2, 2, 2, 2, 2, 3, 2, 3,
                  3, 3, 3, 4, 3, 4, 4, 4, 5, 5],

    "House_Age": [25, 20, 18, 15, 12, 10, 8, 7, 15, 10,
                  8, 6, 5, 4, 10, 7, 5, 3, 2, 1],

    "House_Price": [180000, 195000, 220000, 250000, 270000,
                    295000, 320000, 350000, 360000, 390000,
                    410000, 440000, 470000, 500000, 520000,
                    550000, 580000, 610000, 650000, 690000]
}

df = pd.DataFrame(data)


# ==========================================
# 3. Load and display the dataset
# ==========================================

print("House Price Dataset:")
print(df)

print("\nDataset Information:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())


# ==========================================
# 4. Explore and visualize the data
# ==========================================

# House Area vs House Price
plt.scatter(df["House_Area"], df["House_Price"])
plt.xlabel("House Area")
plt.ylabel("House Price")
plt.title("House Area vs House Price")
plt.show()


# Bedrooms vs House Price
plt.scatter(df["Bedrooms"], df["House_Price"])
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price")
plt.title("Bedrooms vs House Price")
plt.show()


# Bathrooms vs House Price
plt.scatter(df["Bathrooms"], df["House_Price"])
plt.xlabel("Number of Bathrooms")
plt.ylabel("House Price")
plt.title("Bathrooms vs House Price")
plt.show()


# House Age vs House Price
plt.scatter(df["House_Age"], df["House_Price"])
plt.xlabel("House Age")
plt.ylabel("House Price")
plt.title("House Age vs House Price")
plt.show()


# ==========================================
# 5. Select features and target
# ==========================================

X = df[["House_Area", "Bedrooms", "Bathrooms", "House_Age"]]

y = df["House_Price"]


# ==========================================
# 6. Split into training and testing data
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)


# ==========================================
# 7. Train Linear Regression model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)


# ==========================================
# 8. Predict house prices
# ==========================================

y_pred = model.predict(X_test)

print("\nActual vs Predicted Prices:")

for actual, predicted in zip(y_test, y_pred):
    print("Actual:", actual, "Predicted:", round(predicted, 2))


# ==========================================
# 9. Evaluate the model
# ==========================================

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)


# ==========================================
# 10. Plot Actual vs Predicted Prices
# ==========================================

plt.scatter(y_test, y_pred)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

# Perfect prediction line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)

plt.show()


# ==========================================
# 11. Analyze model coefficients
# ==========================================

print("\nModel Coefficients:")

for feature, coefficient in zip(X.columns, model.coef_):
    print(feature, ":", coefficient)

print("\nIntercept:", model.intercept_)


# Find strongest feature
coefficients = pd.Series(model.coef_, index=X.columns)

strongest_feature = coefficients.abs().idxmax()

print("\nFeature with the strongest effect:")
print(strongest_feature)