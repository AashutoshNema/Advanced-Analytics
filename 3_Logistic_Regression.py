# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 03:15:03 2019

@author: aashu
"""

import pandas as pd
import numpy as np
import AdvancedAnalytics

#Importing DataSet
data = pd.read_excel('sonar3by5.xlsx')

#Spliting 
x = data.iloc[:,:-1].values
y = data.iloc[:,60].values

#replacing outliers to Nan in test data
x = pd.DataFrame(x)
x = x.mask(x < 0, 'NaN', axis = 0) #Lowerlimit
x = x.mask(x > 1, 'NaN', axis = 0) ##Upperlimit

#Imputing missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x)
x = imputer.transform(x)
x = pd.DataFrame(x)

#Converting response variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Spliting the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(x_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from AdvancedAnalytics import logreg
print('The following is the Regression Statistics')
logreg.display_binary_split_metrics(classifier, x_train, y_train, x_test, y_test)

# Printing first 15 predicted values
print("\nFirst 15 predicted values\n",y_pred[0:14])

print("\nThe first 15 variables after imputation and prediction\n",x_train[0:15]) 