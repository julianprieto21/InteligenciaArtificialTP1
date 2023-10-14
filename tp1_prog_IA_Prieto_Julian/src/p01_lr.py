import util
import numpy as np
import matplotlib.pyplot as plt
from util import plot, plot_points


def escalar(X):
    media = np.mean(X)
    std = np.std(X)
    X_escalada = (X - media) / std
    return X_escalada


def graficar(x, y, theta=None):
    plt.scatter(x[:, 1], x[:, 2], c=y)
    if not theta is None:
        margin1 = (max(x[:, -2]) - min(x[:, -2])) * 0.2
        margin2 = (max(x[:, -1]) - min(x[:, -1])) * 0.2
        x1 = np.arange(min(x[:, -2]) - margin1, max(x[:, -2]) + margin1, 0.01)
        x2 = -(theta[0] / theta[2] + theta[1] / theta[2] * x1)
        plt.plot(x1, x2, c="red", linewidth=2)
        plt.xlim(x[:, -2].min() - margin1, x[:, -2].max() + margin1)
        plt.ylim(x[:, -1].min() - margin2, x[:, -1].max() + margin2)
    plt.show()


def ruido_gauss_x(x):
    ruido = np.random.normal(0, 0.1, x.shape)
    return x + ruido


def ruido_gauss_y(y):
    ruido = np.random.normal(0, 0.1, y.shape)
    y_ruido = y + ruido
    for i in range(y_ruido.shape[0]):
        if y_ruido[i] >= 0:
            y_ruido[i] = 1
        elif y_ruido[i] < 0:
            y_ruido[i] = -1

    return y_ruido


def calc_grad(X, Y, theta):
    """Calcula el gradiente de la pérdida con respecto a tita."""
    m, n = X.shape

    lambda_value = 0.0225

    margins = Y * X.dot(theta)
    probs = 1.0 / (1 + np.exp(margins))
    grad = -(1.0 / m) * (X.T.dot(probs * Y))  # + lambda_value
    # grad = - 1 / m * (probs * Y) @ X

    return grad


def logistic_regression(X, Y):
    """Entrena un modelo de regresión logística."""
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10
    i = 0

    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta - learning_rate * grad

        if i % 10000 == 0:
            # print('Terminadas %d iteraciones' % i)
            print(f"ERROR: {np.linalg.norm(prev_theta - theta)}")
            print("-------------------------------------")
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print("Convergencia en %d iteraciones" % i)
            print(f"ERROR: {np.linalg.norm(prev_theta - theta)}")
            break
        # learning_rate = 1 / i**2
    return theta


def main():
    print("==== Entrenando modelo en dataset A ====")
    Xa, Ya = util.load_csv("data/ds1_a.csv", add_intercept=True)
    # Xa = ruido_gauss_x(Xa)
    # Ya = ruido_gauss_y(Ya)
    theta = logistic_regression(Xa, Ya)

    # Xa = escalar(Xa)
    graficar(Xa, Ya, theta)

    print("\n==== Entrenando modelo en dataset B ====")
    Xb, Yb = util.load_csv("data/ds1_b.csv", add_intercept=True)
    # Xb = ruido_gauss_x(Xb)
    # Yb = ruido_gauss_y(Yb)
    theta = logistic_regression(Xb, Yb)

    # Xb = escalar(Xb)
    graficar(Xb, Yb, theta)


if __name__ == "__main__":
    main()
