# Importing necessary libraries for data analysis and machine learning
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from scipy import stats
from mlxtend.evaluate import bias_variance_decomp
import markdown
import time

# Setting filename to store the output analysis in a Markdown format with a timestamp
filename = "Output-" + time.strftime("%Y%m%d-%H%M%S") + ".md"

# Open file to write results
with open(filename, "w") as file:
    # Load dataset from CSV file
    df = pd.read_csv('data_cars.csv')

    # Writing basic statistical analysis of the dataset to the file
    file.write("## Basic Statistics Analysis\n")
    file.write(df.describe().to_markdown() + "\n\n")

    # Selecting numeric columns and calculating correlation and covariance, and writing the results to the file
    numeric_cols = df.select_dtypes(include=[np.number])
    file.write("## Numeric Columns Correlation\n")
    file.write(numeric_cols.corr().to_markdown() + "\n\n")
    file.write("## Numeric Columns Covariance\n")
    file.write(numeric_cols.cov().to_markdown() + "\n\n")

    # Checking for null values and writing the result to the file
    file.write("## Null Values\n")
    file.write(df.isnull().sum().to_markdown() + "\n\n") 

    # Fill missing values in numeric columns with mean of respective column
    for col in numeric_cols.columns:
        df[col].fillna(df[col].mean(), inplace=True)

    # Selecting categorical columns and filling missing values with the mode
    categorical_cols = df.select_dtypes(include=[object])
    for col in categorical_cols.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # One-hot encoding of categorical variables
    df = pd.get_dummies(df)

    # Scale the data using MinMaxScaler
    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

    # Removing outliers from scaled data using Z-score
    df_scaled = df_scaled[(np.abs(stats.zscore(df_scaled)) < 10).all(axis=1)]
    file.write("## After outlier removal, number of rows: " + str(len(df_scaled)) + "\n")

    # Splitting the data into features and target variable
    X = df_scaled.drop('Price', axis=1)
    y = df_scaled['Price']

    # Check if data is sufficient for model training
    if len(X) > 0 and len(y) > 0:
        # Splitting the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create and train a Linear Regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict the target variable for the test data
        y_pred = model.predict(X_test)

        # Write model results to the file
        file.write("## Model Results\n")
        file.write('### Coefficients:\n' + str(model.coef_) + '\n')
        file.write('### Intercept:\n' + str(model.intercept_) + '\n')
        file.write('### Mean Absolute Error (MAE):\n' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        file.write('### Mean Squared Error (MSE):\n' + str(metrics.mean_squared_error(y_test, y_pred)) + '\n')
        file.write('### Root Mean Squared Error (RMSE):\n' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + '\n')
        file.write('### R2:\n' + str(metrics.r2_score(y_test, y_pred)) + '\n')
    else:
        file.write("## Insufficient data for model training after preprocessing.\n")

    # New model for bias and variance decomposition
    model = LinearRegression()

    # Compute average loss, bias, and variance using bias_variance_decomp function from mlxtend library
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), 
        loss='mse', num_rounds=200, random_seed=1)

    # Write bias and variance results to the file
    file.write("## Bias and Variance\n")
    file.write('### Average expected loss:\n' + str(avg_expected_loss) + '\n')
    file.write('### Average bias:\n' + str(avg_bias) + '\n')
    file.write('### Average variance:\n' + str(avg_var) + '\n')
