# FraudDetection


# Credit Card Fraud Detection Using Logistic Regression

This project implements a machine learning model to detect fraudulent transactions using logistic regression. The dataset used for this project contains anonymized features about credit card transactions, where the target variable indicates whether a transaction is fraudulent.

## Table of Contents
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Dependencies](#dependencies)
- [How to Run](#how-to-run)
- [Results](#results)
- [License](#license)

## Dataset
The dataset is sourced from the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains the following:
- **Features**: Anonymized numerical features (V1, V2, ..., V28) and the transaction amount.
- **Target Variable**: `Class` (0 for legitimate transactions and 1 for fraudulent transactions).

### Class Distribution
The dataset is highly imbalanced:
- Legitimate transactions: 284,315
- Fraudulent transactions: 492

## Project Workflow
1. **Data Preprocessing**:
   - Load the dataset and explore its structure.
   - Handle missing values (if any).
   - Analyze class distribution and transaction amounts for legitimate and fraudulent transactions.

2. **Resampling**:
   - To address class imbalance, a random sample of legitimate transactions is taken to match the number of fraudulent transactions.

3. **Feature Selection**:
   - The target variable (`Class`) is separated from the features.

4. **Model Training**:
   - Split the data into training and testing sets using an 80-20 split with stratification.
   - Train a logistic regression model on the training set.

5. **Evaluation**:
   - Measure model accuracy on both the training and testing sets.

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`

## How to Run
1. Clone the repository and navigate to the project directory.
2. Ensure you have Python 3 installed along with the required dependencies. Install dependencies using:
   ```bash
   pip install numpy pandas scikit-learn
