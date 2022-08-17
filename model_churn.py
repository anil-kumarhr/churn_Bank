# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 23:45:54 2022

@author: anilhr
"""


import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing

churn = pd.read_excel(r'C:\Users\ANILHR\Downloads\CHURNDATA6.xlsx')


churn=churn[['# total debit transactions for S3','total debit amount for S3','total debit amount','total transactions','total debit transactions','total credit amount for S3','# total debit transactions for S2','AGE','Status']]


label_encoder = preprocessing.LabelEncoder()
# b=['CUS_Gender', 'CUS_Marital_Status','TAR_Desc','Status']
# for i in b:
churn['Status']= label_encoder.fit_transform(churn['Status']) 
pd.to_numeric(churn['Status'])


class_count_1, class_count_0 = churn['Status'].value_counts()
print(class_count_0)
class_1 = churn[churn['Status'] == 0]
print(class_1)
class_0 = churn[churn['Status'] == 1]
class_1_under = class_1.sample(class_count_0)

test_under = pd.concat([class_1_under, class_0], axis=0)




x = test_under.iloc[:,0:3]
y = test_under.iloc[:,8:]









from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=123)

model_churn = RandomForestClassifier(class_weight={0:1, 1:1}, n_estimators= 80, min_samples_leaf= 1, max_depth=4)

model_churn.fit(X_train, y_train)

print(model_churn)


# make predictions
expected = y_test
predicted = model_churn.predict(X_test)
# summarize the fit of the model
#Correction
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

import pickle

pickle.dump(model_churn, open("model_churn.pkl", "wb"))

model = pickle.load(open("model_churn.pkl", "rb"))


# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

