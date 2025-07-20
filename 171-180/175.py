# 🧠 Ejercicio 175/200 — Visualización de Word Embeddings preentrenados con PCA y t-SNE
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# Paso 1: Cargar GloVe 50d (solo palabras relevantes)
def cargar_glove(filepath, palabras_objetivo):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            if palabra in palabras_objetivo:
                vector = np.array(valores[1:], dtype="float32")
                embeddings[palabra] = vector
            if len(embeddings) == len(palabras_objetivo):
                break
    return embeddings


# Paso 2: Preparar matriz y etiquetas
def preparar_datos(embeddings_dict):
    palabras = list(embeddings_dict.keys())
    vectores = np.array([embeddings_dict[p] for p in palabras])
    return palabras, vectores


# Paso 3: Visualizar con PCA o t-SNE
def visualizar(vectores, etiquetas, metodo="pca"):
    if metodo == "pca":
        modelo = PCA(n_components=2)
        reducidos = modelo.fit_transform(vectores)
    elif metodo == "tsne":
        modelo = TSNE(n_components=2, random_state=42, perplexity=5, n_iter=1000)
        reducidos = modelo.fit_transform(vectores)
    else:
        raise ValueError("Método desconocido")

    plt.figure(figsize=(8, 6))
    for i, palabra in enumerate(etiquetas):
        x, y = reducidos[i]
        plt.scatter(x, y, color="blue")
        plt.text(x + 0.01, y + 0.01, palabra, fontsize=9)
    plt.title(f"Visualización de embeddings con {metodo.upper()}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -------- MAIN --------
if __name__ == "__main__":
    # Palabras relacionadas con fake news, medios y polarización
    palabras_clave = [
        "truth",
        "lie",
        "hoax",
        "fake",
        "news",
        "trust",
        "media",
        "bias",
        "fact",
        "verify",
        "mislead",
        "propaganda",
        "fraud",
        "politics",
        "reliable",
        "science",
        "fear",
        "belief",
        "authority",
        "evidence",
    ]

    ruta_glove = "glove.6B.50d.txt"
    if not os.path.exists(ruta_glove):
        print(
            "❌ Archivo GloVe no encontrado. Descárgalo y colócalo junto a este script."
        )
        exit()

    emb = cargar_glove(ruta_glove, palabras_clave)
    etiquetas, vectores = preparar_datos(emb)

    print("📊 Generando visualización con PCA...")
    visualizar(vectores, etiquetas, metodo="pca")

    print("📊 Generando visualización con t-SNE...")
    visualizar(vectores, etiquetas, metodo="tsne")
