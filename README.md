import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Leer el archivo csv

df = pd.read_csv('kc_house_data.csv')
x = df['sqft_living'].values
y = df['price'].values
