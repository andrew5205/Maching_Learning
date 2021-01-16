

# pd.read_csv()
# pd.read_excel()



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
# [:]       indicate Range, w/o providing edge means ALL
# [:-1]     all but not last one, Python range include starting point, but excluding end point
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# # *********************************************************************************************************
# print(dataset)
# #    Country   Age   Salary Purchased
# # 0   France  44.0  72000.0        No
# # 1    Spain  27.0  48000.0       Yes
# # 2  Germany  30.0  54000.0        No
# # 3    Spain  38.0  61000.0        No
# # 4  Germany  40.0      NaN       Yes
# # 5   France  35.0  58000.0       Yes
# # 6    Spain   NaN  52000.0        No
# # 7   France  48.0  79000.0       Yes
# # 8  Germany  50.0  83000.0        No
# # 9   France  37.0  67000.0       Yes

# print(X)
# # ['France' 44.0 72000.0]
# #  ['Spain' 27.0 48000.0]
# #  ['Germany' 30.0 54000.0]
# #  ['Spain' 38.0 61000.0]
# #  ['Germany' 40.0 nan]
# #  ['France' 35.0 58000.0]
# #  ['Spain' nan 52000.0]
# #  ['France' 48.0 79000.0]
# #  ['Germany' 50.0 83000.0]
# #  ['France' 37.0 67000.0]
# # ]

# print(y)
# # ['No' 'Yes' 'No' 'No' 'Yes' 'Yes' 'No' 'Yes' 'No' 'Yes']


# *********************************************************************************************************
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
















