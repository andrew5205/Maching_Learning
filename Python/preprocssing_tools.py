# Data Preprocessing Tools


# 1. Taking care of missing data - SimpleImputer()

# 2.1 Encoding categorical data - ColumnTransformer()
# 2.2 Encoding the Dependent Variable - LabelEncoder()

# 3. Splitting the dataset into the Training set and Test set - train_test_split()

# 4. Feature Scaling - StandardScaler()


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# *********************************************************************************************************
print(dataset)
#    Country   Age   Salary Purchased
# 0   France  44.0  72000.0        No
# 1    Spain  27.0  48000.0       Yes
# 2  Germany  30.0  54000.0        No
# 3    Spain  38.0  61000.0        No
# 4  Germany  40.0      NaN       Yes
# 5   France  35.0  58000.0       Yes
# 6    Spain   NaN  52000.0        No
# 7   France  48.0  79000.0       Yes
# 8  Germany  50.0  83000.0        No
# 9   France  37.0  67000.0       Yes



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

print(X)
# [['France' 44.0 72000.0]
#  ['Spain' 27.0 48000.0]
#  ['Germany' 30.0 54000.0]
#  ['Spain' 38.0 61000.0]
#  ['Germany' 40.0 63777.77777777778]
#  ['France' 35.0 58000.0]
#  ['Spain' 38.77777777777778 52000.0]
#  ['France' 48.0 79000.0]
#  ['Germany' 50.0 83000.0]
#  ['France' 37.0 67000.0]
# ]



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

print(X)
# [[1.0 0.0 0.0 44.0 72000.0]
#  [0.0 0.0 1.0 27.0 48000.0]
#  [0.0 1.0 0.0 30.0 54000.0]
#  [0.0 0.0 1.0 38.0 61000.0]
#  [0.0 1.0 0.0 40.0 63777.77777777778]
#  [1.0 0.0 0.0 35.0 58000.0]
#  [0.0 0.0 1.0 38.77777777777778 52000.0]
#  [1.0 0.0 0.0 48.0 79000.0]
#  [0.0 1.0 0.0 50.0 83000.0]
#  [1.0 0.0 0.0 37.0 67000.0]]


# *********************************************************************************************************
# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)            # [0 1 0 0 1 1 0 1 0 1]



# *********************************************************************************************************
# feature scaling AFTER split into training set + test set
# *********************************************************************************************************


# *********************************************************************************************************
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

# train_test_split(arrays)
# training_set 80%, test_set 20%

# random_state = 1; equal result of train/ test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
print(X_train)
# [[0.0 0.0 1.0 38.77777777777778 52000.0]
#  [0.0 1.0 0.0 40.0 63777.77777777778]
#  [1.0 0.0 0.0 44.0 72000.0]
#  [0.0 0.0 1.0 38.0 61000.0]
#  [0.0 0.0 1.0 27.0 48000.0]
#  [1.0 0.0 0.0 48.0 79000.0]
#  [0.0 1.0 0.0 50.0 83000.0]
#  [1.0 0.0 0.0 35.0 58000.0]
# ]

# random pick?! 
print(X_test)
# [[0.0 1.0 0.0 30.0 54000.0]
#  [1.0 0.0 0.0 37.0 67000.0]
# ]

# correspond to X_train
print(y_train)          # [0 1 0 0 1 1 0 1]

# correspond to X_test
print(y_test)           # [0 1]



# *********************************************************************************************************
# # Standardisation:
# X_stand = X - mean(X)/ (standard deviation(X))

# # Normalisation:
# X_norm = X - min(X)/ (max(X) - min(X))
# *********************************************************************************************************

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# [:, 3:] skip the dummy first 3 col which is binary present as country 
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(X_train)
# [[0.0 0.0 1.0 -0.19159184384578545 -1.0781259408412425]
#  [0.0 1.0 0.0 -0.014117293757057777 -0.07013167641635372]
#  [1.0 0.0 0.0 0.566708506533324 0.633562432710455]
#  [0.0 0.0 1.0 -0.30453019390224867 -0.30786617274297867]
#  [0.0 0.0 1.0 -1.9018011447007988 -1.420463615551582]
#  [1.0 0.0 0.0 1.1475343068237058 1.232653363453549]
#  [0.0 1.0 0.0 1.4379472069688968 1.5749910381638885]
#  [1.0 0.0 0.0 -0.7401495441200351 -0.5646194287757332]
# ]

print(X_test)
# [[0.0 1.0 0.0 -1.4661817944830124 -0.9069571034860727]
#  [1.0 0.0 0.0 -0.44973664397484414 0.2056403393225306]
# ]




