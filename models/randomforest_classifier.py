#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.dates as mdates

def randomforest_classifier(dataframe, currencies=None, include_sp500=True, lag=1, train_size=0.75, trees=30, max_depth=10, leafes=10):
    """
   Trains a Randomforest Classifier on financial data to predict binary outcomes and evaluates its performance.

   This function processes financial data, which can include currency pairs and optionally S&P 500 data, and applies a Decision Tree Classifier for binary classification. It calculates the model's accuracy and plots cumulative returns for comparison.

   Parameters:
   - dataframe (pd.DataFrame): The dataset containing the financial data. It should have a 'DATE' column, currency data columns, and optionally an 'SP500' column.
   - currencies (list of str, optional): List of currency columns to include in the analysis. If None, all available currencies in the dataframe will be used.
   - include_sp500 (bool): Determines whether to include the S&P 500 data in the analysis. Defaults to True.
   - lag (int): The number of periods by which to lag the response variable for prediction. Defaults to 1.
   - train_size (float): The proportion of the dataset to use for training the model. The rest will be used for testing. Defaults to 0.75.
   - trees (int): The number of trees in the forest. More trees can lead to a more robust and stable model, but it also comes with increased computational cost. Default is 30.
   - max_depth (int): The maximum depth of one decision tree in the forest. Helps to control the complexity of the model. Defaults to 10.
   - leafes (int): The maximum number of terminal nodes / leaves in one tree. Helps to control the complexity of the model. Defaults to 10.

   Returns:
   - tuple: A tuple containing a DataFrame of model accuracies and another DataFrame with the cumulative returns for both the 'Long' strategy and the Decision Tree Classifier model. It also generates a plot comparing these returns.

   The function prints the accuracy of the model and displays a plot of the cumulative returns. The accuracies DataFrame includes 'in sample' and 'out of sample' accuracy of the Decision Tree Classifier.
   """

    # Selecting columns based on currencies if provided


    if currencies is not None:
        unavailable_currencies = [currency for currency in currencies if currency not in dataframe.columns]
        if unavailable_currencies:
            available_currencies = ', '.join([col for col in dataframe.columns if col not in ['DATE', 'SP500']])
            unavailable_currencies_str = ', '.join(unavailable_currencies)
            print(
                f"Sorry, {unavailable_currencies_str} is not an available currency pair. Please choose currency pairs from: {available_currencies}")
            return None

    # Setting up response and regressor variables
    y1 = dataframe.iloc[lag:, -1]  # Assuming the last column is the response variable
    y_dates = dataframe.iloc[lag:, 0]
    X1 = dataframe.iloc[:-lag, 1:-1]  # Excluding Date and response variable

    # Including or excluding SP500 based on the flag
    if include_sp500:
        x1_spx = dataframe.iloc[:-lag, -1]
        X1 = pd.concat([X1, x1_spx], axis=1)

    # Calculating binary response
    y1_exp = np.exp(y1) - 1
    y1_binary = (y1_exp > 0).astype(int)

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X1, y1_binary, random_state=42, shuffle=False,
                                                        test_size=1 - train_size)

    # Setting up dates
    y_test_dates = y_dates[y_test.index[0] - lag:]

    # Training the Randomforest Classifier
    randomforest_clf = RandomForestClassifier(n_estimators=trees, random_state=42, max_depth=max_depth, max_leaf_nodes=leafes)
    randomforest_clf.fit(X_train, y_train)
    y_pred_train_rfc = randomforest_clf.predict(X_train)
    y_pred_test_rfc = randomforest_clf.predict(X_test)

    # Changing all 0 to -1 for return calculation
    y_pred_test_rfc[np.where(y_pred_test_rfc == 0)] = -1

    # Calculating returns
    y1_ret = y1_exp
    y_bench = y1_ret[y_test.index[0] - lag:]
    y_long = np.cumsum(y_bench)
    y_rfc = np.cumsum(y_bench * y_pred_test_rfc)

    # Calculating accuracies
    accuracies = pd.DataFrame({
        "Classifiers": ["DTC"],
        "in sample": [accuracy_score(y_train, y_pred_train_rfc)],
        "out of sample": [accuracy_score(y_test, y_pred_test_rfc)]
    }).set_index('Classifiers')

    print(accuracies)

    # Plotting
    plt.figure(figsize=(14, 6))
    test_perform = pd.DataFrame({"Long": y_long, "DTC": y_rfc})
    plt.plot(y_test_dates, test_perform["Long"], "r", label="Long")
    plt.plot(y_test_dates, test_perform["DTC"], "g", label="RFC")
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.legend()
    plt.title("Financial Performance Prediction" + (" with " if include_sp500 else " without ") + "S&P500")
    plt.xlabel('Year')
    plt.ylabel('CumSum Return')
    plt.savefig('classifier.svg', dpi=1000)
    plt.show()

    return accuracies, test_perform
