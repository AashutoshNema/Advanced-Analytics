# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 03:14:44 2019

@author: aashu
"""
#Importing Libraries
import pandas as pd
import numpy as np
from AdvancedAnalytics import ReplaceImputeEncode
from AdvancedAnalytics import NeuralNetwork
import sklearn
from sklearn.neural_network import MLPClassifier
#Reading Data File
df = pd.read_excel("C:/Users/aashu/Desktop/Study/656/Python/CreditHistory_Clean(2).xlsx")

#Changing variable to string
df['checking']=df['checking'].astype(str)
df['coapp']=df['coapp'].astype(str)
df['depends']=df['depends'].astype(str)
df['employed']=df['employed'].astype(str)
df['existcr']=df['existcr'].astype(str)
df['history']=df['history'].astype(str)
df['foreign']=df['foreign'].astype(str)
df['good_bad']=df['good_bad'].astype(str)
df['installp']=df['installp'].astype(str)
df['job']=df['job'].astype(str)
df['marital']=df['marital'].astype(str)
df['other']=df['other'].astype(str)
df['property']=df['property'].astype(str)
df['purpose']=df['purpose'].astype(str)
df['resident']=df['resident'].astype(str)
df['savings']=df['savings'].astype(str)
df['telephon']=df['telephon'].astype(str)
df['housing']=df['housing'].astype(str)

#Creating data map
attribute_map = {
        'age':['I', (19, 120)],
        'amount':['I', (0, 20000)],
        'checking':['N',('1','2','3','4')],
        'coapp':['N',('1', '2', '3')],
        'depends':['B',('1', '2')],
        'duration':['I',(1,72)],
        'employed':['N',('1','2','3','4','5')],
        'existcr':['N',('1','2','3','4')],
        'foreign':['B', ('1','2')],
        'good_bad':['B', ('bad','good')],
        'history':['N', ('0','1','2','3','4')],
        'housing':['N', ('1','2','3')],
        'installp':['N', ('1','2','3','4')] ,
        'job':['N', ('1','2','3','4')] ,
        'marital':['N', ('1','2','3','4')] ,
        'other':['N', ('1','2','3')] ,
        'property':['N', ('1','2','3','4')] ,
        'purpose':['N', ('0','1','2','3','4','5','6','8','9','X')] ,
        'resident':['N', ('1','2','3','4')] ,    
        'savings':['N', ('1','2','3','4','5')] ,    
        'telephon':['B', ('1','2')] }

#Encoding Data
rie = ReplaceImputeEncode(data_map=attribute_map,display=True,drop = False)
encoded_df = rie.fit_transform(df)
encoded_df.dtypes

#Adding encoded values & Removing multiple rows
final_table =  pd.concat([encoded_df,pd.get_dummies(encoded_df['checking'], prefix='checking'),
                         pd.get_dummies(encoded_df['coapp'], prefix='coapp'),
                         pd.get_dummies(encoded_df['depends'], prefix='depends'),
                         pd.get_dummies(encoded_df['employed'], prefix='employed'),
                         pd.get_dummies(encoded_df['good_bad'], prefix='good_bad'),
                         pd.get_dummies(encoded_df['existcr'], prefix='existcr'),
                         pd.get_dummies(encoded_df['foreign'], prefix='foreign'),
                         pd.get_dummies(encoded_df['history'], prefix='history'),
                         pd.get_dummies(encoded_df['housing'], prefix='housing'),
                         pd.get_dummies(encoded_df['installp'], prefix='insatllp'),
                         pd.get_dummies(encoded_df['job'], prefix='job'),
                         pd.get_dummies(encoded_df['other'], prefix='other'),
                         pd.get_dummies(encoded_df['property'], prefix='property'),
                         pd.get_dummies(encoded_df['purpose'], prefix='purpose'),
                         pd.get_dummies(encoded_df['resident'], prefix='resident'),
                         pd.get_dummies(encoded_df['savings'], prefix='savings'),
                         pd.get_dummies(encoded_df['telephon'], prefix='telephon')],axis=1)
final_table.drop(['checking'],axis=1, inplace=True)
final_table.drop(['coapp'],axis=1, inplace=True)
final_table.drop(['depends'],axis=1, inplace=True)
final_table.drop(['employed'],axis=1, inplace=True)
final_table.drop(['existcr'],axis=1, inplace=True)
final_table.drop(['foreign'],axis=1, inplace=True)
final_table.drop(['history'],axis=1, inplace=True)
final_table.drop(['housing'],axis=1, inplace=True)
final_table.drop(['installp'],axis=1, inplace=True)
final_table.drop(['job'],axis=1, inplace=True)
final_table.drop(['other'],axis=1, inplace=True)
final_table.drop(['property'],axis=1, inplace=True)
final_table.drop(['purpose'],axis=1, inplace=True)
final_table.drop(['resident'],axis=1, inplace=True)
final_table.drop(['savings'],axis=1, inplace=True)
final_table.drop(['telephon'],axis=1, inplace=True)

final_table.drop(['good_bad'],axis=1, inplace=True)
final_table.drop(['good_bad_0.0'],axis=1, inplace=True)
final_table
# 10 Fold Cross Validation
from sklearn.model_selection import cross_val_score
ntwrk_list = [(3),(11),(5,4),(6,5),(7,6),(8,7)]
score_list = ['accuracy', 'recall', 'precision', 'f1']
var_list = ['good_bad_1.0']
X = np.asarray(final_table.drop('good_bad_1.0', axis=1))
y = np.asarray(final_table[var_list])
np_y = np.ravel(y)
#Neural Networks for cross validation
for nn in ntwrk_list:
    fnn = MLPClassifier(hidden_layer_sizes=nn, activation='relu',solver='lbfgs', max_iter=2000, random_state=12345)
    mean_score = []
    std_score = []
    print("NList=", nn)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        fnn_10 = cross_val_score(fnn, X, np_y, scoring=s, cv=4)
        mean = fnn_10.mean()
        std = fnn_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        
fnn1 = fnn.fit(X, np_y)
NeuralNetwork.display_binary_metrics(fnn1, X, np_y)    
    
#Selcting model 6,5
from sklearn.model_selection import train_test_split
#Creating Test and Train set
X_train, X_validate, y_train, y_validate = train_test_split(X,np_y,test_size = 0.3, random_state=12345)
fnn = MLPClassifier(hidden_layer_sizes=(6,5), activation='tanh', solver='lbfgs', max_iter=1000,random_state=12345)
fnn = fnn.fit(X_train, y_train)
NeuralNetwork.display_binary_split_metrics(fnn,X_train, y_train, X_validate, y_validate)