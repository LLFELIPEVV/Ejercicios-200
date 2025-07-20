# 🎯 Ejercicio 181/200 — Visualización de embeddings con PCA y t-SNE
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Paso 1: Cargar un subconjunto de GloVe
def cargar_glove(ruta, dimension, palabras_objetivo):
    embeddings = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            partes = linea.strip().split()
            palabra = partes[0]
            if palabra in palabras_objetivo:
                vector = np.array(partes[1:], dtype="float32")
                embeddings[palabra] = vector
            if len(embeddings) == len(palabras_objetivo):
                break
    return embeddings


# Palabras clave típicas en noticias falsas y reales
palabras = [
    "fake",
    "hoax",
    "rumor",
    "fraud",
    "propaganda",
    "clickbait",
    "deceive",
    "mislead",
    "truth",
    "fact",
    "evidence",
    "official",
    "confirmed",
    "source",
    "report",
    "journalism",
    "vaccine",
    "covid",
    "election",
    "government",
]

print(f"Cargando {len(palabras)} palabras...")

glove_path = "glove.6B.100d.txt"
embeddings = cargar_glove(glove_path, 100, palabras)

# Validar que se hayan encontrado todas
if len(embeddings) < len(palabras):
    print("⚠️ Algunas palabras no fueron encontradas en el GloVe. Revisa el archivo.")
    exit()

# Paso 2: Extraer vectores y nombres
vectores = np.array([embeddings[p] for p in palabras])
etiquetas = np.array(palabras)

# Paso 3: Reducción con PCA
pca = PCA(n_components=2)
vectores_pca = pca.fit_transform(vectores)

# Paso 4: Reducción con t-SNE
tsne = TSNE(n_components=2, perplexity=5, random_state=42, n_iter=1000)
vectores_tsne = tsne.fit_transform(vectores)


# Paso 5: Función para graficar resultados
def graficar(vectores_2d, etiquetas, metodo):
    plt.figure(figsize=(10, 6))
    for i, etiqueta in enumerate(etiquetas):
        x, y = vectores_2d[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, etiqueta, fontsize=9)
    plt.title(f"Visualización de embeddings con {metodo}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Mostrar ambos métodos
graficar(vectores_pca, etiquetas, "PCA")
graficar(vectores_tsne, etiquetas, "t-SNE")
