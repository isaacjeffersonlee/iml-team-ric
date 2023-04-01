import pandas as pd
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_validate
from rich.console import Console  # Pretty printing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic

console = Console()
# Load the data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Onehot encode categorical variables
train_df = pd.get_dummies(train_df)
test_df = pd.get_dummies(test_df)

# Dealing with missing values
train_df.dropna(subset=["price_CHF"], inplace=True)
train_df.fillna(train_df.mean(), inplace=True)
test_df.fillna(test_df.mean(), inplace=True)

# Convert into arrays
X_train = train_df.copy()
y_train = X_train.pop("price_CHF")
X_train = np.array(X_train)
X_test = np.array(test_df)
y_train = np.array(y_train)


def fit_and_predict(model):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=5,
        scoring=("r2", "neg_mean_squared_error"),
        return_train_score=True,
    )
    console.log(
        f"5 fold CV Average MSE: {np.mean(scores['test_neg_mean_squared_error'])}"
    )
    console.log(f"Train R2: {np.mean(scores['train_r2'])}")
    print("-" * 45)
    return y_pred


model = KernelRidge(alpha=1.3, kernel="poly", degree=4)
# model = KernelRidge(alpha=1.0, kernel="poly", degree=3)
# for kernel in [DotProduct(), RBF(), Matern(), RationalQuadratic()]:
#     print(f"α: {alpha} | kernel: {kernel}")
#     model = GaussianProcessRegressor(alpha=alpha, kernel=kernel)
#     fit_and_predict(model)

y_pred = fit_and_predict(model)
np.savetxt("./results.csv", y_pred, fmt="%.12f")
