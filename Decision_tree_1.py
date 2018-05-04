
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

#Import data
data = pd.read_csv('data1.csv')
print ("Dataset Length: ", len(data))
print ("Dataset Shape: ", data.shape)

#Splitting
X = data.iloc[:, :-1].values
Y = data.iloc[:, 13].values
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 2/27, random_state = 0)

#Training using Gini
clf_gini = DecisionTreeClassifier(criterion = "gini",random_state = 0,max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

print("Results Using Gini Index:")
print("Actual values:")
print(y_test)
y_pred_gini = clf_gini.predict(X_test)
print("Predicted values:")
print(y_pred_gini)
print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred_gini))  
print ("Accuracy : ",
    accuracy_score(y_test,y_pred_gini)*100)

#Training using Entropy
clf_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 0,max_depth=3, min_samples_leaf=5)
clf_entropy.fit(X_train, y_train)

print("Results Using Entropy Index:")
print("Actual values:")
print(y_test)
y_pred_entropy = clf_entropy.predict(X_test)
print("Predicted values:")
print(y_pred_entropy)
print("Confusion Matrix: ",
        confusion_matrix(y_test, y_pred_entropy))  
print ("Accuracy : ",
    accuracy_score(y_test,y_pred_entropy)*100)
