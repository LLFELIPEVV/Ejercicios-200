# 🧠 Ejercicio 178/200 — Usar y visualizar embeddings GloVe con PCA
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Paso 1: Definir ruta del archivo GloVe (descargado manualmente)
RUTA_GLOVE = "glove.6B.50d.txt"

# Paso 2: Palabras clave a analizar
palabras_clave = [
    "fake",
    "truth",
    "news",
    "government",
    "hoax",
    "virus",
    "president",
    "freedom",
    "vaccine",
    "bleach",
    "pandemic",
    "media",
    "conspiracy",
    "election",
]


# Paso 3: Cargar solo los vectores que nos interesan
def cargar_embeddings_glove(ruta_archivo, palabras_objetivo):
    vectores = {}
    with open(ruta_archivo, "r", encoding="utf-8") as archivo:
        for linea in archivo:
            valores = linea.strip().split()
            palabra = valores[0]
            if palabra in palabras_objetivo:
                vector = np.asarray(valores[1:], dtype="float32")
                vectores[palabra] = vector
            if len(vectores) == len(palabras_objetivo):
                break
    return vectores


# Paso 4: Aplicar PCA para reducir a 2 dimensiones
def reducir_con_pca(vectores_dict):
    palabras = list(vectores_dict.keys())
    vectores = np.array([vectores_dict[p] for p in palabras])

    pca = PCA(n_components=2)
    vectores_reducidos = pca.fit_transform(vectores)

    return palabras, vectores_reducidos


# Paso 5: Graficar en 2D
def graficar_embeddings(palabras, vectores_2d):
    plt.figure(figsize=(10, 8))
    for palabra, coord in zip(palabras, vectores_2d):
        x, y = coord
        plt.scatter(x, y)
        plt.annotate(palabra, (x + 0.01, y + 0.01), fontsize=10)
    plt.title("Embeddings GloVe (50d) reducidos con PCA")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("embeddings_glove_pca.png")
    plt.show()


# Pipeline principal
if __name__ == "__main__":
    vectores = cargar_embeddings_glove(RUTA_GLOVE, palabras_clave)
    palabras, vectores_reducidos = reducir_con_pca(vectores)
    graficar_embeddings(palabras, vectores_reducidos)
