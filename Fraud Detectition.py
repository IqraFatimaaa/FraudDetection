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

# Load dataset
file_path = r"C:/Users/PMLS/Desktop/creditcard.csv"
credit_card_data = pd.read_csv(file_path)

# Display dataset overview
print("Dataset Head:")
print(credit_card_data.head())
print("\nDataset Info:")
credit_card_data.info()

# Check for missing values
missing_values = credit_card_data.isnull().sum()
print("\nMissing Values in Each Column:")
print(missing_values)

# Analyze class distribution
class_distribution = credit_card_data['Class'].value_counts()
print("\nClass Distribution:")
print(class_distribution)

# Separate legitimate and fraudulent transactions
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]

# Summary statistics for transaction amounts
print("\nLegitimate Transactions Summary:")
print(legit.Amount.describe())
print("\nFraudulent Transactions Summary:")
print(fraud.Amount.describe())

# Sampling to balance the dataset
sample_size = fraud.shape[0]
legit_sample = legit.sample(n=sample_size, random_state=42)

# Combine the sampled data
balanced_dataset = pd.concat([legit_sample, fraud], axis=0)
print("\nBalanced Dataset Class Distribution:")
print(balanced_dataset['Class'].value_counts())

# Prepare features and target variable
X = balanced_dataset.drop(columns='Class', axis=1)
Y = balanced_dataset['Class']

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
print("\nDataset Shapes:")
print(f"Features: {X.shape}, Training Set: {X_train.shape}, Test Set: {X_test.shape}")

# Logistic regression model training
model = LogisticRegression(max_iter=200, solver='liblinear')
model.fit(X_train, Y_train)

# Evaluate the model
X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)
print("\nTraining Data Accuracy:", training_accuracy)

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Test Data Accuracy:", test_accuracy)
