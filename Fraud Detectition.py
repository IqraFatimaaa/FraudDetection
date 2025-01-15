# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 23:30:14 2025

@author: PMYLS
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

credit_card_data = pd.read_csv(r"C:/Users/PMLS/Desktop/creditcard.csv")

credit_card_data.head()
credit_card_data.tail()
credit_card_data.info()

credit_card_data.isnull().sum() 
credit_card_data['Class'].value_counts()
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

print(legit.shape)
print(fraud.shape)
legit.Amount.describe()
fraud.Amount.describe()
credit_card_data.groupby('Class').mean()
legit_sample = legit.sample(n=492)


new_dataset = pd.concat([legit_sample, fraud], axis=0)


new_dataset.tail()
new_dataset['Class'].value_counts()
new_dataset.groupby('Class').mean()

X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print(X.shape, X_train.shape, X_test.shape)

model = LogisticRegression()
model.fit(X_train, Y_train)
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print('Accuracy score on Test Data : ', test_data_accuracy)