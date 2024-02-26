import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


df = pd.read_csv('kc_house_data.csv')


# Outlier Removal

outliers = df.quantile(q=0.97, numeric_only=True)  # calculates the 97th percentile (quantile) for each numeric column in the DataFrame

# remove rows where the values exceed the 97th percentile threshold defined by 'outliers'
df = df[(df['price'] < outliers['price'])]
df = df[(df['bedrooms'] < outliers['bedrooms'])]
df = df[(df['bathrooms'] < outliers['bathrooms'])]
df = df[(df['sqft_living'] < outliers['sqft_living'])]


# Feature Engineering

df['zipcode'] = df['zipcode'].astype('category') # convert the 'zipcode' column to categorical type

import datetime

df['age'] = datetime.datetime.now().year - df['yr_built']

df['yr_renovated'] = np.where(df['yr_renovated'] > 0, 1, 0)
df['sqft_basement'] = np.where(df['sqft_basement'] > 0, 1, 0)

x = df.drop(['id', 'date', 'lat', 'long', 'price'], axis=1)
y = df[['price']]

x = pd.get_dummies(x, drop_first=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=42)


# Modelling

def algo_test(x, y):
    # Defining all the models
    L = LinearRegression()
    R = Ridge()
    Lass = Lasso()
    E = ElasticNet()
    ETR = ExtraTreeRegressor()
    GBR = GradientBoostingRegressor()
    kn = KNeighborsRegressor()
    dt = DecisionTreeRegressor()
    xgb = XGBRegressor()

    algos = [L, R, Lass, E, ETR, GBR, kn, dt, xgb]
    algo_names = [
        'Linear', 'Ridge', 'Lasso', 'ElasticNet', 'Extra Tree',
        'Gradient Boosting', 'KNeighborsRegressor', 'Decision Tree',
        'XGBRegressor'
    ]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2, random_state=42)

    r_squared = []
    rmse = []
    mae = []

    # Creating a DataFrame to store accuracy and error metrics
    result = pd.DataFrame(columns=['R_Squared', 'RMSE', 'MAE'],
                          index=algo_names)

    for algo in algos:
        p = algo.fit(x_train, y_train).predict(x_test)
        r_squared.append(r2_score(y_test, p))
        rmse.append(mean_squared_error(y_test, p)**.5)
        mae.append(mean_absolute_error(y_test, p))

    # Populating the result DataFrame with accuracy and error metrics
    result.R_Squared = r_squared
    result.RMSE = rmse
    result.MAE = mae

    # Sorting the result table based on R-squared (accuracy) score
    rtable = result.sort_values('R_Squared', ascending=False)
    return rtable

print(algo_test(x, y))