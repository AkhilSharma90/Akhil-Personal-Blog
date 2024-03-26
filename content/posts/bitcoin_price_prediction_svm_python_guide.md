+++
title = 'Bitcoin Price Prediction A Detailed Guide with Support Vector Machine in Python'
date = 2024-02-02T18:31:22+05:30
draft = false
+++


The allure of predicting Bitcoin prices lies in the volatile nature of the cryptocurrency market. With the right analytical tools and algorithms, we can glean insights into future trends, helping investors make informed decisions. This blog post dissects a Python project that employs a Support Vector Machine (SVM) to forecast Bitcoin prices, offering a granular look at the programming and statistical concepts underpinning this predictive model.

## Connecting and Accessing Data

First and foremost, accessing the historical Bitcoin price data is crucial:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Mounting the Google Drive in Colab links our coding environment directly to our dataset repository. This connection is vital for seamless data access and manipulation, ensuring that our Python scripts can read and process the Bitcoin price data stored in a CSV file on the drive.

## Essential Libraries for Machine Learning

The foundation of any data analysis project rests on the libraries and tools utilized:

1. **Data Manipulation and Analysis**:

   ```python
   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split
   ```

   - `pandas` and `numpy` are indispensable for data wrangling—reading, cleaning, and transforming data to prepare it for analysis.
   - `train_test_split` from `sklearn` is a function that simplifies the division of data into training and testing sets, ensuring that our model can learn from a portion of the data and then be evaluated on a separate set to validate its predictive accuracy.

2. **Predictive Modeling**:

   ```python
   from sklearn.svm import SVR
   ```

   - The `SVR` (Support Vector Regression) module from `sklearn` is chosen for its capability to handle non-linear data patterns, a common characteristic of financial markets, especially in cryptocurrencies like Bitcoin.

## In-depth Data Preprocessing

The next phase involves preparing our data for the predictive model:

```python
data = pd.read_csv('/content/drive/My Drive/bitcoin.csv')
data.drop(['Date'], 1, inplace=True)
```

Reading the dataset with `pandas` allows for a straightforward manipulation process. We remove the `Date` column, focusing solely on price values, as the time component is not needed for this specific prediction model.

We then set up our target variable for prediction:

```python
predictionDays = 30
data['Prediction'] = data[['Price']].shift(-predictionDays)
```

By shifting the `Price` column by 30 days, we create a future-looking `Prediction` column, which becomes the target for our machine learning model.

The dataset is split into features (`x`) and target (`y`):

```python
x = np.array(data.drop(['Prediction'], 1))
x = x[:len(data)-predictionDays]

y = np.array(data['Prediction'])
y = y[:-predictionDays]
```

This separation is critical for training the model, where `x` contains the input features (prices) and `y` holds the future prices we aim to predict.

## Detailed Model Training and Testing

After preprocessing, the data is divided for training and testing:

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

A balanced split ensures the model is not biased towards the data it has seen during training, allowing for an accurate evaluation of its predictive performance.

The SVM model is then configured and trained:

```python
SVR_RBF = SVR(kernel='rbf', C=1e3, gamma=0.00001)
SVR_RBF.fit(x_train, y_train)
```

The choice of the Radial Basis Function (RBF) kernel is deliberate, providing the SVM with the ability to handle the complex patterns and volatility of Bitcoin prices. Parameters `C` and `gamma` are tuned to balance the model's complexity and its capability to generalize on unseen data.

We then assess the model's performance:

```python
print(f"Performance of Support Vector Machine (Regression) using radial basis function: {str(SVR_RBF.score(x_test, y_test))}")
```

The `score` function offers a quick glimpse into the model's effectiveness, quantifying its accuracy in predicting the test data.

## Comprehensive Prediction Analysis

Finally, we employ the model to forecast future prices:

```python
print("PREDICTED PRICE for 30 days")
svm_prediction = SVR_RBF.predict(predictionDays_array)
print(svm_prediction)

print("REAL PRICE for 30 days")
print(data.tail(predictionDays))
```

This section not only reveals the model's predictions for the next 30 days but also juxtaposes these forecasts with the actual prices, providing a stark comparison that highlights the model's predictive prowess or potential shortcomings.
