# ✅ Ejercicio 64/200 — Análisis de clústeres en representaciones vectoriales de texto
# Objetivo: Visualizar y comparar agrupamientos semánticos usando GloVe y Embedding entrenado

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Input, Embedding, GlobalAveragePooling1D, TextVectorization

# ============================== 📦 CARGA Y PREPROCESAMIENTO ==============================

# Cargar datasets de noticias reales y falsas
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

# Asignar etiquetas binarias
fake["label"] = 0
true["label"] = 1

# Unificar datasets y eliminar filas vacías
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

# Extraer textos y etiquetas
X = df["text"].values
y = df["label"].values

# Dividir datos (estratificado) para asegurar balance entre clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ============================== 🔠 REPRESENTACIÓN GloVe PROMEDIO ==============================


def cargar_embeddings(path):
    """Carga embeddings GloVe desde archivo .txt"""
    embeddings = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    return embeddings


def texto_a_vector(texto, embeddings, dim=100):
    """Convierte texto a vector promedio de palabras conocidas"""
    palabras = texto.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    return np.mean(vectores, axis=0) if vectores else np.zeros(dim)


# Cargar GloVe y generar representaciones vectoriales para X_test
embedding_index = cargar_embeddings("Gloove/glove.6B.100d.txt")
glove_vectors = np.array([texto_a_vector(t, embedding_index) for t in X_test])

# ============================== 🔠 REPRESENTACIÓN CON EMBEDDING ENTRENADO ==============================

# Vectorizar texto con secuencias enteras
vectorizador = TextVectorization(
    max_tokens=10000, output_mode="int", output_sequence_length=300
)
vectorizador.adapt(X_train)

X_test_seq = vectorizador(X_test)

# Modelo minimalista para obtener embeddings entrenables
model_embedding = Sequential(
    [
        Input(shape=(300,)),
        Embedding(input_dim=10000, output_dim=100),  # Entrenado desde cero
        GlobalAveragePooling1D(),  # Promedia la secuencia embebida
    ]
)

# Usamos el modelo solo para generar vectores, no entrenamos
embedding_vectors = model_embedding.predict(X_test_seq)

# ============================== 🎨 REDUCCIÓN Y CLUSTERING ==============================


def reducir_dim(vectors, method="tsne"):
    """Reduce dimensionalidad a 2D para visualización"""
    if method == "tsne":
        return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(
            vectors
        )
    return PCA(n_components=2).fit_transform(vectors)


def graficar(vectors_2d, etiquetas, title):
    """Visualiza vectores en 2D coloreados por etiquetas"""
    plt.figure(figsize=(8, 6))
    plt.scatter(
        vectors_2d[:, 0], vectors_2d[:, 1], c=etiquetas, cmap="coolwarm", alpha=0.6
    )
    plt.title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar(label="Etiqueta o clúster")
    plt.grid(True)
    plt.show()


# Agrupamiento con KMeans (2 clústeres: fake vs real)
kmeans_glove = KMeans(n_clusters=2, random_state=42)
clusters_glove = kmeans_glove.fit_predict(glove_vectors)

kmeans_embed = KMeans(n_clusters=2, random_state=42)
clusters_embed = kmeans_embed.fit_predict(embedding_vectors)

# Silhouette score: evalúa calidad de agrupamiento
print("🔹 Silhouette GloVe:", silhouette_score(glove_vectors, clusters_glove))
print("🔹 Silhouette Embedding:", silhouette_score(embedding_vectors, clusters_embed))

# Visualización final
graficar(
    reducir_dim(glove_vectors), clusters_glove, "Clustering KMeans — GloVe promedio"
)
graficar(
    reducir_dim(embedding_vectors),
    clusters_embed,
    "Clustering KMeans — Embedding entrenado",
)
