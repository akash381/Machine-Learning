#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 03:14:04 2018

@author: akash
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('data.txt', sep='\s+')
X = dataset.iloc[:, 2:7].values
y = dataset.iloc[:, 7:8].values
# Label Encoding
'''from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X1 = LabelEncoder()
X[:, 3] = labelencoder_X1.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X2 = LabelEncoder()
X[:, 9] = labelencoder_X2.fit_transform(X[:, 9])
onehotencoder = OneHotEncoder(categorical_features = [9])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X3 = LabelEncoder()
X[:, 11] = labelencoder_X3.fit_transform(X[:, 11])
onehotencoder = OneHotEncoder(categorical_features = [11])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X4 = LabelEncoder()
X[:, 15] = labelencoder_X4.fit_transform(X[:, 15])
onehotencoder = OneHotEncoder(categorical_features = [15])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X5 = LabelEncoder()
X[:, 19] = labelencoder_X5.fit_transform(X[:, 19])
onehotencoder = OneHotEncoder(categorical_features = [19])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X6 = LabelEncoder()
X[:, 23] = labelencoder_X6.fit_transform(X[:, 23])
onehotencoder = OneHotEncoder(categorical_features = [23])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X7 = LabelEncoder()
X[:, 25] = labelencoder_X7.fit_transform(X[:, 25])
onehotencoder = OneHotEncoder(categorical_features = [25])
X = onehotencoder.fit_transform(X).toarray()
'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' ,strategy = 'mean', axis = 0)
imputer = imputer.fit(y)
y = imputer.transform(y)

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' ,strategy = 'mean', axis = 0)
imputer = imputer.fit(X)
X = imputer.transform(X)

# Splitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .000005, random_state = 0)


# Fitting to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)


'''y_pred_approx = (y_pred>1.5).astype(int)
y_test_approx = (y_test>1.5).astype(int)
y_test = y_test.astype(np.float)
from sklearn.metrics import accuracy_score
print ("Accuracy : ",accuracy_score((y_test>1.5).astype(int),(y_pred>1.5).astype(int))*100)
'''
# Plotting residual errors in training data
plt.style.use('fivethirtyeight')
plt.scatter(regressor.predict(X_train), regressor.predict(X_train) - y_train,
            color = "green", s = 10, label = 'Train data')
plt.scatter(regressor.predict(X_test), regressor.predict(X_test) - y_test,
            color = "blue", s = 10, label = 'Test data')
plt.hlines(y = 0, xmin = 2, xmax = 3, linewidth = 2)
plt.legend(loc = 'upper right')
plt.title("Residual errors")
plt.show()

'''
#Backward Elimination SL = 0.05
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((270, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 19, 20, 21, 22, 23, 25, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 19, 21, 22, 23, 25, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 21, 22, 23, 25, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 21, 22, 23, 25, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 21, 22, 23, 25, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 21, 22, 23, 25, 27, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 21, 22, 23, 25, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 2, 6, 7, 8, 9, 10, 11, 12, 16, 17, 21, 23, 25, 28]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
'''