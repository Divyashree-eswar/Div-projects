# -*- coding: utf-8 -*-
"""
Simple Linear regression model implementation
"""
#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,1]

#spliting the data into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =1/3, random_state =0)

#building the linear regression model
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

#performing the prediction on the test set 
y_pred = reg.predict(X_test)

#plotting the regressison line.
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, reg.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()