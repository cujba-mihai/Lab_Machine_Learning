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

# File name
filename = "Output-" + time.strftime("%Y%m%d-%H%M%S") + ".md"

with open(filename, "w") as file:
    # Reading data from CSV file
    df = pd.read_csv('data_cars.csv')

    # Basic statistics analysis
    file.write("## Basic Statistics Analysis\n")
    file.write(df.describe().to_markdown() + "\n\n")

    numeric_cols = df.select_dtypes(include=[np.number])
    file.write("## Numeric Columns Correlation\n")
    file.write(numeric_cols.corr().to_markdown() + "\n\n")
    file.write("## Numeric Columns Covariance\n")
    file.write(numeric_cols.cov().to_markdown() + "\n\n")

    # sns.pairplot(df)  # This command produces a plot which can't be represented in a markdown file

    file.write("## Null Values\n")
    file.write(df.isnull().sum().to_markdown() + "\n\n") 

    for col in numeric_cols.columns:
        df[col].fillna(df[col].mean(), inplace=True)

    categorical_cols = df.select_dtypes(include=[object])
    for col in categorical_cols.columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    df = pd.get_dummies(df)

    min_max_scaler = preprocessing.MinMaxScaler()
    df_scaled = pd.DataFrame(min_max_scaler.fit_transform(df), columns=df.columns)

    file.write("## After scaling, number of rows: " + str(len(df_scaled)) + "\n")
    df_scaled = df_scaled[(np.abs(stats.zscore(df_scaled)) < 10).all(axis=1)] # increased Z-score threshold to 10
    file.write("## After outlier removal, number of rows: " + str(len(df_scaled)) + "\n")

    X = df_scaled.drop('Price', axis=1)
    y = df_scaled['Price']

    if len(X) > 0 and len(y) > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        file.write("## Model Results\n")
        file.write('### Coefficients:\n' + str(model.coef_) + '\n')
        file.write('### Intercept:\n' + str(model.intercept_) + '\n')
        file.write('### Mean Absolute Error (MAE):\n' + str(metrics.mean_absolute_error(y_test, y_pred)) + '\n')
        file.write('### Mean Squared Error (MSE):\n' + str(metrics.mean_squared_error(y_test, y_pred)) + '\n')
        file.write('### Root Mean Squared Error (RMSE):\n' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred))) + '\n')
        file.write('### R2:\n' + str(metrics.r2_score(y_test, y_pred)) + '\n')
    else:
        file.write("## Insufficient data for model training after preprocessing.\n")

    model = LinearRegression()

    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(
        model, np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), 
        loss='mse', num_rounds=200, random_seed=1)

    file.write("## Bias and Variance\n")
    file.write('### Average expected loss:\n' + str(avg_expected_loss) + '\n')
    file.write('### Average bias:\n' + str(avg_bias) + '\n')
    file.write('### Average variance:\n' + str(avg_var) + '\n')
