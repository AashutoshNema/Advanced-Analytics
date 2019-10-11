# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 22:55:40 2019

@author: aashu
"""
#Getting Library
from imblearn.under_sampling import RandomUnderSampler
from AdvancedAnalytics import ReplaceImputeEncode, logreg, calculate
import math
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier

get_ipython().system('pip install imblearn')

#Importing FIle
file_path = 'C:/Users/aashu/Desktop/Study/656/Python/'
df = pd.read_excel(file_path+"CreditData_RareEvent.xlsx")
df.head()
df.describe()

#Changing Data Types
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
df['resident']=df['resident'].astype(str)
df['savings']=df['savings'].astype(str)
df['telephon']=df['telephon'].astype(str)
df['housing']=df['housing'].astype(str)

#Creating attribute map
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
'resident':['N', ('1','2','3','4')] ,    
'savings':['N', ('1','2','3','4','5')] ,    
'telephon':['B', ('1','2')] }

#ReplaceImputeEncode
rie=ReplaceImputeEncode(data_map=attribute_map, nominal_encoding='one-hot',interval_scale='std', drop=True, display=False)
encoded_df = rie.fit_transform(df)

# Creating X and y, numpy arrays
# bad=0 and good=1
y = np.asarray(encoded_df['good_bad']) # The target is not scaled or imputed
X = np.asarray(encoded_df.drop('good_bad',axis=1))

# Setup false positive and false negative costs for each transaction
fp_cost = np.array(df['amount'])
fn_cost = np.array(0.15*df['amount'])

#Function for calculating loss and confusion matrix
def binary_loss(y, y_predict, fp_cost, fn_cost, display=True):
    loss = [0, 0] #False Neg Cost, False Pos Cost
    conf_mat = [0, 0, 0, 0] #tn, fp, fn, tp
    for j in range(len(y)):
        if y[j]==0:
            if y_predict[j]==0:
                conf_mat[0] += 1 #True Negative
            else:
                conf_mat[1] += 1 #False Positive
                loss[1] += fp_cost[j]
        else:
            if y_predict[j]==1:
                conf_mat[3] += 1 #True Positive
            else:
                conf_mat[2] += 1 #False Negative
                loss[0] += fn_cost[j]
    if display:
        fn_loss = loss[0]
        fp_loss = loss[1]
        total_loss = fn_loss + fp_loss
        misc = conf_mat[1] + conf_mat[2]
        misc = misc/len(y)
        print("{:.<23s}{:10.4f}".format("Misclassification Rate", misc))
        print("{:.<23s}{:10.0f}".format("False Negative Loss", fn_loss))
        print("{:.<23s}{:10.0f}".format("False Positive Loss", fp_loss))
        print("{:.<23s}{:10.0f}".format("Total Loss", total_loss))
    return loss, conf_mat


#Decision Tree
#10 fold CV with variation of depth
from sklearn.model_selection import cross_val_score
score_list = ['accuracy', 'recall', 'precision', 'f1']
search_depths = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
for d in search_depths:
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=d, min_samples_split=5, min_samples_leaf=5)
    mean_score = []
    std_score = []
    print("max_depth=", d)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        dtc_10 = cross_val_score(dtc, X, y, scoring=s, cv=10)
        mean = dtc_10.mean()
        std = dtc_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        
# Setup 20 random number seeds for use in creating random samples
np.random.seed(12345)
max_seed = 2**10 - 1
rand_val = np.random.randint(1, high=max_seed, size=20)
# Ratios of Majority:Minority Events
ratio = [ '50:50', '60:40', '75:25', '80:20', '85:15' ]
# Dictionaries contains number of minority and majority
# events in each ratio sample where n_majority = ratio x n_minority
rus_ratio = ({0:500, 1:500}, {0:500, 1:750}, {0:500, 1:1167},{0:500, 1:2000}, {0:500, 1:4500})

d=2
# Best model is one that minimizes the loss
c_list = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3, 4, 1e+64]
min_loss = 1e64
best_ratio = 0
for k in range(len(rus_ratio)):
    print("\nDesicion Tree Model using " + ratio[k] + " RUS")
    best_c = 0
    min_loss_c = 1e64
    for j in range(len(c_list)):
        c = c_list[j]
        fn_loss = np.zeros(len(rand_val))
        fp_loss = np.zeros(len(rand_val))
        misc = np.zeros(len(rand_val))
        for i in range(len(rand_val)):
            rus = RandomUnderSampler(ratio=rus_ratio[k],random_state=rand_val[i],return_indices=False,replacement=False)
            X_rus, y_rus = rus.fit_sample(X, y)
            dtc = DecisionTreeClassifier(criterion='gini', max_depth=d, min_samples_split=5, min_samples_leaf=5)
            dtc.fit(X_rus,y_rus)
            loss, conf_mat = calculate.binary_loss(y,dtc.predict(X),fp_cost, fn_cost, display=False)
            fn_loss[i] = loss[0]
            fp_loss[i] = loss[1]
            misc[i] = (conf_mat[1] + conf_mat[2])/y.shape[0]
        avg_misc = np.average(misc)
        t_loss = fp_loss+fn_loss
        avg_loss = np.average(t_loss)
        if avg_loss < min_loss_c:
            min_loss_c = avg_loss
            se_loss_c = np.std(t_loss)/math.sqrt(len(rand_val))
            best_c = c
            misc_c = avg_misc
            fn_avg_loss = np.average(fn_loss)
            fp_avg_loss = np.average(fp_loss)
    if min_loss_c < min_loss:
        min_loss = min_loss_c
        se_loss = se_loss_c
        best_ratio = k
        best_reg = best_c
    print("{:.<23s}{:12.2E}".format("Best C", best_c))
    print("{:.<23s}{:12.4f}".format("Misclassification Rate",misc_c))
    print("{:.<23s} ${:10,.0f}".format("False Negative Loss",fn_avg_loss))
    print("{:.<23s} ${:10,.0f}".format("False Positive Loss",fp_avg_loss))
    print("{:.<23s} ${:10,.0f}{:5s}${:<,.0f}".format("Total Loss",min_loss_c, " +/- ", se_loss_c))
print("")
print("{:.<23s}{:>12s}".format("Best RUS Ratio", ratio[best_ratio]))
print("{:.<23s}{:12.2E}".format("Best C", best_reg))
print("{:.<23s} ${:10,.0f}{:5s}${:<,.0f}".format("Lowest Loss", min_loss, " +/-", se_loss))

#Ensemble Modeling - Averaging Classification Probabilities
n_obs = len(y)
n_rand = 100
predicted_prob = np.zeros((n_obs,n_rand))
avg_prob = np.zeros(n_obs)

# Setup 100 random number seeds for use in creating random samples
np.random.seed(12345)
max_seed = 2**10 - 1
rand_value = np.random.randint(1, high=max_seed, size=n_rand)

# Model 100 random samples, each with a 70:30 ratio
for i in range(len(rand_value)):
    rus = RandomUnderSampler(ratio=rus_ratio[best_ratio],random_state=rand_value[i], return_indices=False,replacement=False)
    X_rus, y_rus = rus.fit_sample(X, y)
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=d, min_samples_split=5, min_samples_leaf=5)
    dtc.fit(X_rus,y_rus)
    predicted_prob[0:n_obs, i] = dtc.predict_proba(X)[0:n_obs, 0]

for i in range(n_obs):
    avg_prob[i] = np.mean(predicted_prob[i,0:n_rand])
    
# Set y_pred equal to the predicted classification
y_pred = avg_prob[0:n_obs] < 0.5
y_pred.astype(np.int)

# Calculate loss from using the ensemble predictions
print("\nEnsemble Estimates based on averaging",len(rand_value), "Models")
loss, conf_mat = calculate.binary_loss(y, y_pred, fp_cost, fn_cost)
