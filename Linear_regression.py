
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, 7:8].values
y = dataset.iloc[:, 13].values

# Splitting
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Fitting to Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction
y_pred = regressor.predict(X_test)

y_pred_approx = (y_pred>1.5).astype(int)
y_test_approx = (y_test>1.5).astype(int)
y_test = y_test.astype(np.float)
from sklearn.metrics import accuracy_score
print ("Accuracy : ",accuracy_score((y_test>1.5).astype(int),(y_pred>1.5).astype(int))*100)

# Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Heart Beat vs Heart Disease  (Training set)')
plt.xlabel('Heart_Beat')
plt.ylabel('Heart_Disease')
plt.show()

# Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Heart Beat vs Heart Disease (Test set)')
plt.xlabel('Heart_Beat')
plt.ylabel('Heart_Disease')
plt.show()