# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.model_selection import cross_val_score

kernels = [DotProduct, RBF, Matern, RationalQuadratic]

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

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

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """
    """scores = []
    for kernel in kernels:
        kernel_score = []
        for alpha in [0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]:
            gpr = GaussianProcessRegressor(alpha=alpha, kernel=kernel())
            reg = gpr.fit(X_train, y_train)
            kernel_score.append(cross_val_score(reg, X_train, y_train, cv=5).mean())  
        scores.append(kernel_score)
    
    for score in scores:
        print(score)
        print(np.mean(score))"""

    gpr = GaussianProcessRegressor(alpha=0.09, kernel=Matern())
    gpr.fit(X_train, y_train)
    y_pred = gpr.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred) 
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

