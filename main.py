import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def compute_cost(X, y, theta):
    m = len(y)
    predictions = np.dot(X, theta)
    cost = np.sum((predictions - y) ** 2) / (2 * m)
    return cost


# Leer el archivo csv
df = pd.read_csv("kc_house_data.csv")
x = df["sqft_living"].values
y = df["price"].values

# Define the range of k values
k_range = range(1, 11)

# Loop over different k values
for k in k_range:

    X = []
    for xi in x:
        X.append([xi**g for g in range(0, k + 1)])
    X = np.array(X)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True)

    loss = []
    for train_idx, val_idx in kf.split(X):

        # Dividir el dataset en data de entreno y test
        X_train, y_train = X[train_idx], y[train_idx]  # variables de entreno
        X_val, y_val = X[val_idx], y[val_idx]  # variables de testeo

        # Normalizar los datos de entrada
        X_train = (X_train - np.mean(X_train)) / np.std(X_train)
        X_val = (X_val - np.mean(X_val)) / np.std(X_val)

        # Definir los parámetros del modelo y la tasa de aprendizaje
        theta = np.random.randn(1, k + 1)[0]  # pesos aleatorios
        lr = 0.0001  # tasa de aprendizaje
        n_iterations = 1000  # número máximo de iteraciones

        # Ejecutar el descenso del gradiente para encontrar los coeficientes del modelo
        for iteration in range(n_iterations):
            gradient = (-2 * np.dot(X_train.T, y_train)) / len(
                y_train
            )  # obtener el gradiente
            theta = theta - lr * gradient  #
            cost = compute_cost(X_train, y_train, theta)

        # Obtener la perdida y la almacenamos
        L = (np.transpose((np.subtract(y_val, X_val.dot(theta))))).dot(
            np.subtract(y_val, X_val.dot(theta))
        )
        L = L * (1 / len(X_val))
        loss.append(L)

    # Operar la media de todas las perdidas almacenadas
    print(f"k = {k}, loss = {np.mean(loss)}")
    # Encontrar el valor de k con la menor perdida promedio
best_k = np.argmin(loss)
print("Best k value:", best_k)
