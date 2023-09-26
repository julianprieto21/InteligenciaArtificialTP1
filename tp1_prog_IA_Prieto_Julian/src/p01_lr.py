import util
import numpy as np
import matplotlib.pyplot as plt

costos_A = []
costos_B = []


def calc_grad(X, Y, theta):
    """Calcula el gradiente de la pérdida con respecto a tita."""
    m, n = X.shape

    norma_2 = 0
    for i in range(len(theta)):
        norma_2 += theta[i] ** 2

    lambda_value = 0.0225

    margins = Y * X.dot(theta)
    probs = 1.0 / (1 + np.exp(margins))
    grad = -(1.0 / m) * (X.T.dot(probs * Y)) + lambda_value

    # lambda_value = 0.001
    # # Agregar término de regularización L2
    # grad_regularization = (lambda_value / m) * theta
    # grad_regularization[0] = 0  # No regularizar el término de sesgo (intercept)
    # grad += grad_regularization

    return grad


def logistic_regression(X, Y, costos):
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

        # costos.append(np.linalg.norm(prev_theta - theta))

        if i % 10000 == 0:
            # print('Terminadas %d iteraciones' % i)
            print(f"ERROR: {np.linalg.norm(prev_theta - theta)}")
            print("-------------------------------------")
        if np.linalg.norm(prev_theta - theta) < 1e-15:
            print("Convergencia en %d iteraciones" % i)
            print(f"ERROR: {np.linalg.norm(prev_theta - theta)}")
            break

    return


def main():
    print("==== Entrenando modelo en dataset A ====")
    Xa, Ya = util.load_csv("data/ds1_a.csv", add_intercept=True)
    logistic_regression(Xa, Ya, costos_A)

    # plt.clf()
    # plt.plot(costos_A)
    # plt.show()

    print("\n==== Entrenando modelo en dataset B ====")
    Xb, Yb = util.load_csv("data/ds1_b.csv", add_intercept=True)
    logistic_regression(Xb, Yb, costos_B)

    # plt.clf()
    # plt.plot(costos_B)
    # plt.show()


if __name__ == "__main__":
    main()
