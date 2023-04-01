import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Plotting and Visualizing data
from sklearn.linear_model import LinearRegression


# data
data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')

y_train = data_train.iloc[:, 1]
x_train = data_train.iloc[:, 2:]
x_test = data_test.iloc[:, 1:]

# regression
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)

# ouput
df = pd.DataFrame({'Id': data_test.iloc[:, 0],
                   'y': y_pred})
df.to_csv('result.csv', index=False)
