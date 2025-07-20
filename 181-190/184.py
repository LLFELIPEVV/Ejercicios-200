# 🎯 Ejercicio 184/200 — Visualización de Embeddings GloVe con PCA y t-SNE
import os
import re
import html
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.layers import TextVectorization


# 1. Sanitizar texto de ejemplo
def sanitizar(texto):
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip().lower()
    return texto


# 2. Cargar vectores GloVe
def cargar_glove(filepath, vocab_tokenizer, dimension=50):
    embeddings_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            valores = line.split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings_index[palabra] = vector

    palabras_objetivo = list(vocab_tokenizer.word_index.keys())[
        :200
    ]  # máx 200 palabras
    palabras_encontradas = []
    vectores = []

    for palabra in palabras_objetivo:
        vec = embeddings_index.get(palabra)
        if vec is not None:
            palabras_encontradas.append(palabra)
            vectores.append(vec)

    return palabras_encontradas, np.array(vectores)


# 3. Visualizar con PCA y t-SNE
def visualizar(vocabulario, vectores):
    print("Reduciendo con PCA...")
    pca = PCA(n_components=2)
    emb_pca = pca.fit_transform(vectores)

    print("Reduciendo con t-SNE...")
    tsne = TSNE(n_components=2, perplexity=25, init="pca", n_iter=1000)
    emb_tsne = tsne.fit_transform(vectores)

    plt.figure(figsize=(14, 6))

    # Gráfico PCA
    plt.subplot(1, 2, 1)
    plt.title("PCA - Embeddings")
    for i, palabra in enumerate(vocabulario):
        x, y = emb_pca[i]
        plt.scatter(x, y, color="blue", s=10)
        plt.text(x + 0.01, y + 0.01, palabra, fontsize=8)

    # Gráfico t-SNE
    plt.subplot(1, 2, 2)
    plt.title("t-SNE - Embeddings")
    for i, palabra in enumerate(vocabulario):
        x, y = emb_tsne[i]
        plt.scatter(x, y, color="green", s=10)
        plt.text(x + 0.01, y + 0.01, palabra, fontsize=8)

    plt.tight_layout()
    plt.savefig("embeddings_visualizados.png")
    plt.show()


# 4. Script principal
if __name__ == "__main__":
    print("=== Visualización de GloVe con PCA y t-SNE ===\n")

    # Corpus de ejemplo para extraer vocabulario representativo
    corpus = [
        "The president held a press conference today",
        "Scientists discovered a new exoplanet in the galaxy",
        "Click here to claim your free iPhone",
        "COVID-19 vaccine is effective according to the WHO",
        "Aliens are controlling the government says anonymous source",
        "Climate change effects are visible worldwide",
        "Government denies involvement in leaked documents",
        "BREAKING: secret agency develops mind control chip",
        "AI and Machine Learning are transforming industries",
        "You won a million dollars in a secret lottery",
    ]
    corpus = [sanitizar(t) for t in corpus]

    # Tokenizar corpus para seleccionar palabras más frecuentes
    tokenizer = TextVectorization(max_tokens=1000)
    tokenizer.adapt(corpus)

    ruta_glove = "glove.6B.50d.txt"
    if not os.path.exists(ruta_glove):
        print("❌ ERROR: No se encuentra el archivo GloVe.")
        print("Descárgalo desde: https://nlp.stanford.edu/data/glove.6B.zip")
        exit()

    palabras, vectores = cargar_glove(ruta_glove, tokenizer)
    print(f"Palabras visualizadas: {len(palabras)}\n")
    visualizar(palabras, vectores)
