# Data Preprocessing Tools

# Taking care of missing data
# Encoding categorical data - ColumnTransformer()
# Encoding the Dependent Variable - LabelEncoder()



# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
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



# *********************************************************************************************************
# Taking care of missing data
from sklearn.impute import SimpleImputer

# replace np.nan by MEAN value of that column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# fit(location) method will looking from given location, X[:, 1:3] => col1 to col2 
imputer.fit(X[:, 1:3])
# transform(location) method will do the replacement job
# transform() - return the new updated object
# update the dataset
X[:, 1:3] = imputer.transform(X[:, 1:3])

# print(X)
# # [['France' 44.0 72000.0]
# #  ['Spain' 27.0 48000.0]
# #  ['Germany' 30.0 54000.0]
# #  ['Spain' 38.0 61000.0]
# #  ['Germany' 40.0 63777.77777777778]
# #  ['France' 35.0 58000.0]
# #  ['Spain' 38.77777777777778 52000.0]
# #  ['France' 48.0 79000.0]
# #  ['Germany' 50.0 83000.0]
# #  ['France' 37.0 67000.0]
# # ]



# *********************************************************************************************************
# Encoding categorical data - into binary
# Encoding the Independent Variable
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
# country column transform to binary
# ColumnTransformer(['encoder', OneHotEncoder(), [column index]])
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# fitting transforming, force in np array
X = np.array(ct.fit_transform(X))

# print(X)
# # [[1.0 0.0 0.0 44.0 72000.0]
# #  [0.0 0.0 1.0 27.0 48000.0]
# #  [0.0 1.0 0.0 30.0 54000.0]
# #  [0.0 0.0 1.0 38.0 61000.0]
# #  [0.0 1.0 0.0 40.0 63777.77777777778]
# #  [1.0 0.0 0.0 35.0 58000.0]
# #  [0.0 0.0 1.0 38.77777777777778 52000.0]
# #  [1.0 0.0 0.0 48.0 79000.0]
# #  [0.0 1.0 0.0 50.0 83000.0]
# #  [1.0 0.0 0.0 37.0 67000.0]]


# *********************************************************************************************************
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
# print(y)            # [0 1 0 0 1 1 0 1 0 1]










