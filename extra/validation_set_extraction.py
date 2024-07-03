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

y_test.to_csv('../data/support/diamonds_y_val.csv', sep =';')
x_test.to_csv('../data/support/diamonds_x_val.csv', sep =';')