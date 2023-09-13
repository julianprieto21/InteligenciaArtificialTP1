import csv

import matplotlib.pyplot as plt
import numpy as np
import json


def add_intercept_fn(x):
    """Agrega término independiente a la matriz x.

    Entrada:
        x: 2D NumPy array.

    Salida:
        Nueva matriz igual a x con 1's in la columna 0.
    """
    new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x

def load_csv(csv_path, label_col='y', add_intercept=False):
    """Carga un CSV.

    Entrada:
         csv_path: directorio al CSV.
         label_col: nombre de la columna a usar como etiqueta (debería ser 'y' o 'l').
         add_intercept: agrega término indepentiente.

    Salida:
        xs: Numpy array de x-valores (inputs).
        ys: Numpy array de y-valores (labels).
    """

    # Cargar encabezados
    with open(csv_path, 'r', newline='') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Cargar features y etiquetas
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    l_cols = [i for i in range(len(headers)) if headers[i] == label_col]
    inputs = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols)
    labels = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=l_cols)

    if inputs.ndim == 1:
        inputs = np.expand_dims(inputs, -1)

    if add_intercept:
        inputs = add_intercept_fn(inputs)

    return inputs, labels


def plot(x, y, theta, save_path, correction=1.0):
    """Grafica conjunto de datos y parámetros de regresión logística ajustados.

    Entrada:
        x: matriz de datos, uno por fila.
        y: Vector de etiquetas en el {0, 1}.
        theta: Vector de parammetros.
        save_path: directorio para grabar el gráfico.
        correction: factor de correción (no se usa).
    """
    # Graficar dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Graficar borde de decisión (theta^t x = 0)
    x1 = np.arange(min(x[:, -2]), max(x[:, -2]), 0.01)
    x2 = -(theta[0] / theta[2] * correction + theta[1] / theta[2] * x1)
    plt.plot(x1, x2, c='red', linewidth=2)

    # Agrega etiquetas y graba
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)



def plot_points(x, y):
    """Grafica un scatter plot donde x son los puntos e y la etiqueta"""
    x_one = x[y == 0, :]
    x_two = x[y == 1, :]
    
    plt.scatter(x_one[:,0], x_one[:,1], marker='x', color='red')
    plt.scatter(x_two[:,0], x_two[:,1], marker='o', color='blue')
