import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# Load the data
train_df = pd.read_csv("train.csv")
train_df.dropna(subset=["price_CHF"])
test_df = pd.read_csv("test.csv")

# Onehot encode categorical variables
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# Dealing with missing values
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Convert into arrays
X_train = train_df.copy()
y_train = X_train.pop("price_CHF")
X_train = np.array(X_train)
X_test = np.array(test_df)
y_train = np.array(y_train)

# Define model
model = XGBRegressor(objective="reg:squarederror", missing=1, seed=42)

# Fit model
model.fit(X_train, y_train)

# Inference
y_hat = model.predict(X_test)
np.savetxt("./results.csv", y_hat, fmt="%.12f")
