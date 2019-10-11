# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 02:00:22 2019

@author: aashu
"""
#Importing Library
import pandas as pd
import numpy as np
from AdvancedAnalytics import ReplaceImputeEncode

#Reading Data
df = pd.read_excel("diamondswmissing.xlsx")

#Missing Values
data_map = {\
            'obs': [4,(1,53940)],\
            'Carat':[0,(0.2,5.5)],\
            'cut':[2,('Fair','Good','Premium','Very Good')],\
            'color':[2,('D','E','F','G','H','I','J')],\
            'clarity':[2,('I1','IF','SI1','SI2','VS1','VS2','VVS1','VVS2')],\
            'depth':[0,(40,80)],\
            'table':[0,(40,100)],\
            'x':[0,(0,11)],\
            'y':[0,(0,60)],\
            'z':[0,(0,32)],\
            'price':[0,(300,20000)]\
            }
rie = ReplaceImputeEncode(data_map=data_map,display=True)
df.rie = rie.fit_transform(df)

#Imputing Missing Values
from sklearn import preprocessing
interval_attributes = ['Carat','depth','table','x','y','z']
interval_data = df.as_matrix(columns = interval_attributes)
interval_imputer = preprocessing.Imputer(strategy = 'mean')
imputed_interval_data = interval_imputer.fit_transform(interval_data)

print("Imputed Interval Data:\n", imputed_interval_data)

# Convert String Categorical Attribute to Numbers for further assesment
# Mapping of categories to numbers for attribute 'cut'
cut_map = {'Ideal':0, 'Premium':1, 'Good':2, 'Very Good':3, 'Fair':4}
df['cut'] = df['cut'].map(cut_map)
# Mapping of categories to numbers for attribute 'color'
color_map = {'E':0,'I':1,'J':2,'H':3,'F':4,'G':5,'D':6}
df['color'] = df['color'].map(color_map)
# Mapping of categories to numbers for attribute 'clarity'
clarity_map = {'SI2':0,'SI':1,'VS1':2,'VS2':3,'VVS2':4,'VVS1':5,'I1':6,'IF':7}
df['clarity'] = df['clarity'].map(clarity_map)
print(df)

# Converting nominal data from the dataframe into a numpy array
nominal_attributes = ['cut','color','clarity']
nominal_data = df.as_matrix(columns=nominal_attributes)
# Create Imputer for Categorical Data
cat_imputer = preprocessing.Imputer(strategy='most_frequent')
# Imputing missing values in the Categorical Data
imputed_nominal_data = cat_imputer.fit_transform(nominal_data)
#inserting imputed data in the data frame
df[['cut','color','clarity']] = imputed_nominal_data
df[['Carat','depth','table','x','y','z']] = imputed_interval_data
df.head()

#Encoding
scaler = preprocessing.StandardScaler() # Create an instance of StandardScaler()
scaler.fit(imputed_interval_data)
scaled_interval_data = scaler.transform(imputed_interval_data)
print("Imputed & Scaled Interval Data\n", scaled_interval_data)

# Create an instance of the OneHotEncoder & Selecting Attributes
onehot = preprocessing.OneHotEncoder()
hot_array = onehot.fit_transform(imputed_nominal_data).toarray()
print(hot_array)

from AdvancedAnalytics import linreg
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


y = df['price']
x = df.drop('price',axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

lr=LinearRegression()

col=[]
for i in range(x_train.shape[1]):
    col.append('X'+str(i))

lr.fit(x_train,y_train)
print("\n*** LINEAR REGRESSION ***")
linreg.display_coef(lr, x_train, y_train, col)
linreg.display_metrics(lr, x_train, y_train)

y_hat= lr.predict(x_test)
xtestarr = np.asanyarray(x_test)
ytestarr = np.asanyarray(y_test)

print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y_test) ** 2))

#Maximum, minimum and mean of predicted value
prediction_min = y_hat.min()
print("\nPredicted minimum\n",prediction_min)
prediction_max = y_hat.max()
print("\nPredicted maximum\n",prediction_max)
prediction_mean = y_hat.mean()
print("\nPredicted mean\n",prediction_mean)

#Maximum, minimum and mean of imputed value
actual_min = y_test.idxmin(axis = 0)
print("\nActual minimum\n",actual_min)
actual_max = y_test.idxmax(axis = 0)
print("\nActual maximum\n",actual_max)
actual_mean = y_test.mean(axis = 0)
print("\nActual mean\n",actual_mean)

# Printing first 15 predicted values
print("\nFirst 15 predicted values\n",y_hat[0:14])

# Printing table
final_table = df.head(15)
from pandas import ExcelWriter
writer = ExcelWriter('PythonHW.xlsx')
final_table.to_excel(writer)
writer.save()
