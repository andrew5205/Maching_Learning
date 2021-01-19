

# Linear
# y = b0 + b1 * x1

# y: Dependent Variable (DV)
# x: Independent Variable (IV)
# b1: Coeddicient 
# b0: Constant

# sum( y - y^) ** 2


# *********************************************************************************************************************
# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)




# *********************************************************************************************************************
# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
f1 = plt.figure(1)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
# plt.show()

# Visualising the Test set results
f2 = plt.figure(2)
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'green')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


# # ********************************************************************
# # To show all figure at once 
# # 
# f1 = plt.figure(1)
# # code for f1 
# f2 = plt.figure(2)
# # code for f2 
# plt.show()
# # ********************************************************************



