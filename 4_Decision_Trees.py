# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 00:07:26 2019

@author: aashu
"""
#Importing Libraries
import pandas as pd
import numpy as np
from AdvancedAnalytics import DecisionTree
from AdvancedAnalytics import ReplaceImputeEncode
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import graphviz 
 
#Adding File
df = pd.read_excel("C:/Users/aashu/Desktop/Study/656/Python/CreditHistory_Clean.xlsx")
#Exploring data
print("Credit Data with %i observations & %i attributes.\n"%df.shape, df[0:10])
#Converting data type
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
#Making attribute map
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
#Replace Impute Encode
rie = ReplaceImputeEncode(data_map=attribute_map, display=True,drop = False)
encoded_df = rie.fit_transform(df)
encoded_df.head()
encoded_df.dtypes

final_table =  pd.concat([encoded_df,pd.get_dummies(encoded_df['checking'], prefix='checking'),
                         pd.get_dummies(encoded_df['coapp'], prefix='coapp'),
                         pd.get_dummies(encoded_df['depends'], prefix='depends'),
                         pd.get_dummies(encoded_df['employed'], prefix='employed'),
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
final_table.head()

final_table.columns
response = ['good_bad']
X = np.asarray(final_table.drop(response, axis=1))
y = np.asarray(final_table[response])

#Implementing decision tree
from sklearn.model_selection import cross_val_score
search_depths = [5,6,7,8,10,12,15,20,25]
score_list = ['accuracy', 'recall', 'precision', 'f1']
for d in search_depths:
    decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=d,min_samples_split=5, min_samples_leaf=5)
    mean_score = []
    std_score = []
    print("max_depth=", d)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        decisiontree_10 = cross_val_score(decisiontree, X, y, scoring=s, cv=10)
        mean = decisiontree_10.mean()
        std = decisiontree_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        
#viewing 5 depth results
from sklearn import metrics
decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=5,min_samples_split=5, min_samples_leaf=5)
decisiontree = decisiontree.fit(X,y)
predtree = decisiontree.predict(X)
print("Accuracy of Decision Trees: " , metrics.accuracy_score(y, predtree))

#TrainTest
from sklearn.model_selection import train_test_split
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size = 0.3, random_state=1)
finaltree = DecisionTreeClassifier(criterion='gini', max_depth=5,min_samples_split=5, min_samples_leaf=5)
finaltree = finaltree.fit(X_train,y_train)
decisiontree = DecisionTreeClassifier(criterion='gini', max_depth=5,min_samples_split=5, min_samples_leaf=5)
decisiontree = decisiontree.fit(X_train,y_train)
predtree = finaltree.predict(X_validate)
trainpred=finaltree.predict(X_train)
print("Accuracy of Decision Tree: " , metrics.accuracy_score(y_validate, predtree))
print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
for s in score_list:
        decisiontree_10 = cross_val_score(decisiontree, X, y, scoring=s, cv=10)
        mean = decisiontree_10.mean()
        std = decisiontree_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        
#Printing Tree
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
import matplotlib.image as mpimg
featureNames= final_table.columns[0:68]
dot_data = StringIO()
#featureNames=encoded_df[0:68]
export_graphviz(finaltree,out_file=dot_data, class_names= ['1:Good','0:Bad'], feature_names = featureNames, filled=True, rounded=True,  special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())