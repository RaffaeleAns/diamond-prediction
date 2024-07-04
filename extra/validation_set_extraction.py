"""
Script that extract the validation set from data/diamonds.csv
validation set (x and y) is saved in data/support/
"""

import pandas as pd
from sklearn.model_selection import train_test_split

diamonds = pd.read_csv("../data/diamonds.csv")
diamonds = diamonds[(diamonds.x * diamonds.y * diamonds.z != 0) & (diamonds.price > 0)]
diamonds_processed = diamonds.drop(columns=['depth', 'table', 'y', 'z'])
diamonds_dummy = pd.get_dummies(diamonds_processed, columns=['cut', 'color', 'clarity'], drop_first=True)
x = diamonds_dummy.drop(columns='price')
y = diamonds_dummy.price

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

y_test.to_frame().to_parquet('../data/support/diamonds_regression_y_val.parquet')
x_test.to_parquet('../data/support/diamonds_regression_x_val.parquet')


diamonds_processed_xgb = diamonds.copy()
diamonds_processed_xgb['cut'] = pd.Categorical(diamonds_processed_xgb['cut'], categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'], ordered=True)
diamonds_processed_xgb['color'] = pd.Categorical(diamonds_processed_xgb['color'], categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'], ordered=True)
diamonds_processed_xgb['clarity'] = pd.Categorical(diamonds_processed_xgb['clarity'], categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'], ordered=True)


x_train_xbg, x_test_xbg, y_train_xbg, y_test_xbg = train_test_split(diamonds_processed_xgb.drop(columns='price'), diamonds_processed_xgb['price'], test_size=0.2, random_state=42)

x_test_xbg.to_parquet('../data/support/diamonds_xgboost_x_val.parquet')
y_test_xbg.to_frame().to_parquet('../data/support/diamonds_xgboost_y_val.parquet')