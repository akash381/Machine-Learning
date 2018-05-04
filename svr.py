import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing Dataset 
dataset = pd.read_csv('data1.csv')
X = dataset.iloc[:, 3:4].values
y = dataset.iloc[:, 13:14].values



#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#fitting SVR to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)

#predicting new result
y_pred = regressor.predict(X)

#Visualizing results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()