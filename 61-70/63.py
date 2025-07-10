# ✅ Ejercicio 63/200 — Comparación visual de representaciones de texto: GloVe vs TextVectorization + Embedding
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Input, Embedding, GlobalAveragePooling1D, TextVectorization
from keras.optimizers import Adam

# 📥 Cargar y preparar el dataset
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

fake["label"] = 0  # Etiqueta para noticias falsas
true["label"] = 1  # Etiqueta para noticias reales

# 🧹 Unificar y limpiar el dataframe
df = pd.concat([fake, true], ignore_index=True)
df = df[["text", "label"]].dropna()

X = df["text"].values
y = df["label"].values

# 🔀 Separar en entrenamiento y prueba (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# 📦 Función para cargar embeddings GloVe desde archivo
def cargar_embeddings(path):
    embeddings = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    return embeddings


# 🔠 Convertir texto en vector promedio de embeddings de palabras
def texto_a_vector(texto, embeddings, dim=100):
    palabras = texto.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    return np.mean(vectores, axis=0) if vectores else np.zeros(dim)


# 📥 Obtener vectores GloVe promedio
glove_path = "Gloove/glove.6B.100d.txt"
embedding_index = cargar_embeddings(glove_path)
glove_vectors = np.array([texto_a_vector(t, embedding_index) for t in X_test])

# 🧠 Vectorización + Embedding entrenado
vectorizador = TextVectorization(
    max_tokens=10000,  # Tamaño del vocabulario
    output_mode="int",  # Salida como secuencia de enteros
    output_sequence_length=300,  # Longitud fija de secuencia
)
vectorizador.adapt(X_train)  # Ajustar al vocabulario del texto

X_test_seq = vectorizador(X_test)

# 🔧 Modelo para generar representaciones desde Embedding entrenado
model = Sequential(
    [
        Input(shape=(300,)),  # Secuencia de longitud fija
        Embedding(input_dim=10000, output_dim=100),  # Embedding entrenado desde cero
        GlobalAveragePooling1D(),  # Promediar sobre la secuencia
    ]
)
model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])

# 🧬 Obtener vectores de embedding para comparación
embedding_vectors = model.predict(X_test_seq)


# 🎛️ Función de reducción de dimensionalidad
def reducir_dim(vectors, method="tsne"):
    if method == "tsne":
        return TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(
            vectors
        )
    else:
        return PCA(n_components=2).fit_transform(vectors)


# 📊 Función para visualizar en 2D
def graficar(vectors_2d, y, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], c=y, cmap="coolwarm", alpha=0.6)
    plt.title(title)
    plt.xlabel("Componente 1")
    plt.ylabel("Componente 2")
    plt.colorbar(label="Etiqueta (0=Fake, 1=Real)")
    plt.grid(True)
    plt.show()


# 🔍 Reducir dimensionalidad y graficar comparaciones
glove_2d = reducir_dim(glove_vectors, method="tsne")
graficar(glove_2d, y_test, "Distribución GloVe promedio (TSNE)")

embedding_2d = reducir_dim(embedding_vectors, method="tsne")
graficar(embedding_2d, y_test, "Distribución TextVectorization + Embedding (TSNE)")
