# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 01:57:14 2019

@author: aashu
"""
#Importing Panda
import pandas as pd
import seaborn as sns
#Reading file
print('File Description')
df = "sonar_hw1.csv"
File = pd.read_csv(df)
File.info()
#Finding number of Missing Values
print('Frequency of Missing Values')
for key,value in File.iteritems():
    y=value.isnull().sum(axis=0)
    print(key,y)
#Heat map of Missing values
print('Heat Map of Missing Values')
sns.heatmap(File.isnull(),cbar=False)
# Mean, Max, Min, Meadian of each frequency
print('Max, Min, Median of each frequency')
print(File.describe().unstack())
#Getting frequency for outlier
print('Outlier Frequency')
i = 0
ol = list()
for key,value in File.iteritems():
    for x in value:
        if x>1 or x<0:
            ol.append(x)
            i = i+1
        else: continue
    print(key,i)
    i = 0
#getting list of outliers
print('list of outliers')
print (ol)