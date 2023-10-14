from matplotlib.image import imread, imsave
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from datetime import datetime
import os

large_image = imread("data\peppers-large.tiff")
small_image = imread("data\peppers-small.tiff")


def distancia_euclideana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def k_means(data, k, max_iter=30):
    inicio = datetime.now()

    centroides = data[
        np.random.choice(len(data), k, replace=False)
    ]  # Inicializo los valores de los centroides con pixeles al azar

    for _ in range(max_iter):
        etiquetas = np.array(
            [
                np.argmin([distancia_euclideana(x, c) for c in centroides])
                for x in data
            ]  # Calculo de etiquetas
        )

        new_centroides = np.array(
            [data[etiquetas == i].mean(axis=0) for i in range(k)]
        )  # Calculo de centroides

        # Verificar convergencia
        if np.all(centroides == new_centroides):
            break

        centroides = new_centroides
    print(f"El modelo se entrenó en: {datetime.now() - inicio}")
    return centroides, etiquetas


def comprimir_image(imagen, centroides):
    data = np.array(imagen).reshape(-1, 3)  # Paso la imagen a una matriz de pixeles
    etiquetas = np.ones(len(data), dtype=int)  # Inicializo las etiquetas de los pixeles
    for i in range(len(data)):
        distancia = 999  # Inicializo la distancia más grande
        for j in range(len(centroides)):
            distancia_temp = distancia_euclideana(
                data[i], centroides[j]
            )  # Calculo la distancia entre el pixel y el centroide
            if (
                distancia_temp < distancia
            ):  # Si la distancia es menor que la anterior, actualizo la distancia y la etiqueta
                distancia = distancia_temp
                etiquetas[i] = j

    data_comprimida = centroides[
        etiquetas
    ]  # Obtengo los pixeles de la nueva data comprimida
    imagen_comprimida = data_comprimida.reshape(512, 512, 3).astype(
        np.uint8
    )  # Paso la data comprimida a una nueva imagen
    return imagen_comprimida


data = np.array(small_image).reshape(-1, 3)
k = 16
centroides, etiquetas = k_means(data, k)
new_image = comprimir_image(large_image, centroides)

# imsave("results/peppers-compressed.png", new_image)

plt.subplot(1, 2, 1)
plt.title(
    f"Imagen Original: {os.path.getsize('data/peppers-large.tiff') / 1024:.2f} KB"
)
plt.imshow(small_image)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.title(
    f"Imagen Comprimida: {os.path.getsize('results/peppers-compressed.png') / 1024:.2f} KB"
)
plt.imshow(new_image)
plt.axis("off")
plt.show()

## Calculo del factor de compresión

resolucion_imagen = 512 * 512
bits_per_pixel = 3 * np.log2(256)
bits_per_pixel_comprimida = np.log2(16)

bits_original = resolucion_imagen * bits_per_pixel
bits_comprimida = resolucion_imagen * bits_per_pixel_comprimida

factor = bits_original / bits_comprimida  # = 6
