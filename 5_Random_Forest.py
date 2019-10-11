# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 00:43:08 2019

@author: aashu
"""

#importing libraries
import pandas as pd
import numpy as np
from AdvancedAnalytics import ReplaceImputeEncode
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_validate
from AdvancedAnalytics import DecisionTree
#from sklearn.model_selection import train_test_split
#from AdvancedAnalytics import DecisionTree
import math

#importing data
df = pd.read_excel("OilProduction.xlsx")

#df = pd.DataFrame(df)
#Changing nominal data to string
df['Operator'] = df['Operator'].astype(str)
df['County'] = df['County'].astype(str)
#creating data map
attribute_map = {
        'Log_Cum_Production':['I',(8,15)],
        'Log_Proppant_LB':['I',(6,18)],
        'Log_Carbonate':['I',(-4,4)],
        'Log_Frac_Fluid_GL':['I',(7,18)],
        'Log_GrossPerforatedInterval':['I',(4,9)],
        'Log_LowerPerforation_xy':['I',(8,10)],
        'Log_UpperPerforation_xy':['I',(8,10)],
        'Log_TotalDepth':['I',(8,10)],
        'N_Stages':['I',(2,14)],
        'X_Well':['I',(-100,-95)],
        'Y_Well':['I',(30,35)],
        'Operator':['N',('1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28')],
        'County':['N',('1','2','3','4','5','6','7','8','9','10','11','12','13','14')]
        }
#Encoding data
rie = ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',display=True,interval_scale = None,drop = False)
encoded_df = rie.fit_transform(df)

#from sklearn.preprocessing import OneHotEncoder
#Creating X and y
varlist = ['Log_Cum_Production']
X = encoded_df.drop(varlist, axis=1)
y = encoded_df[varlist]
np_y = np.ravel(y)
col = rie.col
col.remove('Log_Cum_Production')

#Random Forest
rfr = RandomForestRegressor(random_state=12345)
rfr = rfr.fit(X, np_y)
DecisionTree.display_metrics(rfr, X, np_y)
DecisionTree.display_importance(rfr, col)

#Decision Tree
min_mse = 1e64
max_depth = [3,4,5,6,7,8,9,10,11,12,13,14]
score_list = ['neg_mean_squared_error', 'neg_mean_absolute_error']
score_names = ['MSE','MAE']
print("\n****************Decision Tree************")
for d in max_depth:
    print("\nDepth = ",d)
    dtr = DecisionTreeRegressor(max_depth= d, max_features='auto', random_state=12345)
    scores = cross_validate(dtr, X, np_y, scoring=score_list,return_train_score=False, cv=4)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    i=0
    for s in score_list:
        var = "test_"+s
        mean = math.fabs(scores[var].mean())
        std = scores[var].std()
        label = score_names[i]
        i += 1
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(label, mean, std))
        if label == 'MSE' and mean < min_mse:
            min_mse = mean
            best_depth_dt = d
        
print("Best Depth (trees) = ", best_depth_dt)
