from matplotlib.image import imread
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

A_large = imread("data\peppers-large.tiff")
A_small = imread("data\peppers-small.tiff")

# plt.imshow(A)
# plt.show()

# data = A_small[:, :, 2]
# data = A_small.reshape(-1, 3)
# # PCA
# pca = PCA(n_components=2)
# pca.fit(data)
# pca_data = pca.transform(data)

# vars = pca.explained_variance_ratio_
# var1 = round(100 * vars[0], 2)
# var2 = round(100 * vars[1], 2)

# # KMEANS
# cant_clusters = 16
# kmeans = KMeans(n_clusters=cant_clusters)
# kmeans.fit(data)
# y_kmeans = pd.Series(kmeans.predict(data))

# centroides = kmeans.cluster_centers_
# centroides_pca = pca.transform(centroides)

# plt.scatter(pca_data[:, 0], pca_data[:, 1], c=kmeans.labels_, cmap="Paired")
# plt.scatter(centroides_pca[:, 0], centroides_pca[:, 1], c="red", s=200)
# plt.xlabel("Componente principal 1 ({}%)".format(var1))
# plt.ylabel("Componente principal 2 ({}%)".format(var2))
# plt.title("Visualización de clusters por PCA")
# plt.show()

# print(centroides[0])


# # Obtener las dimensiones de la imagen
# alto, ancho, canales = A_small.shape
# # Redimensionar la matriz de píxeles para trabajar con K-Means
# A_reshape = A_small.reshape(-1, canales)
# # Número de clusters
# n_clusters = 16
# # Crear un modelo K-Means
# kmeans = KMeans(n_clusters=n_clusters, n_init=30)
# kmeans.fit(A_reshape)
# # Etiquetas de cluster para cada píxel
# etiquetas = kmeans.labels_
# # Centroides de los clusters
# centroides = kmeans.cluster_centers_
# # Asignar el valor del centroide más cercano a cada píxel
# imagen_comprimida = centroides[etiquetas].reshape(alto, ancho, canales)

# plt.subplot(1, 2, 1)
# plt.title("Imagen Original")
# plt.imshow(A_small)
# plt.axis("off")
# plt.subplot(1, 2, 2)
# plt.title("Imagen Comprimida")
# plt.imshow(imagen_comprimida.astype(np.uint8))
# plt.axis("off")
# plt.show()


def distancia_euclideana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def k_means(data, k, max_iter=30):
    centroides = data[
        np.random.choice(len(data), k, replace=False)
    ]  # Inicializo los valores de los centroides con pixeles al azar

    for _ in range(max_iter):
        etiquetas = np.array(
            [np.argmin([distancia_euclideana(x, c) for c in centroides]) for x in data]
        )

        new_centroides = np.array([data[etiquetas == i].mean(axis=0) for i in range(k)])

        # Verificar convergencia
        if np.all(centroides == new_centroides):
            break

        centroides = new_centroides

    return centroides, etiquetas


data = np.array(A_small).reshape(-1, 3)
k = 16
centroides, etiquetas = k_means(data, k)

data_comprimida = centroides[etiquetas]

imagen_comprimida = data_comprimida.reshape(128, 128, 3).astype(np.uint8)

print(centroides)

# plt.subplot(1, 2, 1)
# plt.title("Imagen Original")
# plt.imshow(A_small)
# plt.axis("off")
# plt.subplot(1, 2, 2)
# plt.title("Imagen Comprimida")
# plt.imshow(imagen_comprimida.astype(np.uint8))
# plt.axis("off")
# plt.show()
