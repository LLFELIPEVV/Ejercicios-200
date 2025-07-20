# 🧠 Ejercicio 171/200 — Embeddings con Keras y visualización con t-SNE
import re
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Embedding, TextVectorization


# Paso 1: Leer frases desde archivo
def cargar_frases(ruta):
    with open(ruta, "r", encoding="utf-8") as f:
        lineas = f.readlines()
    return list(set([l.strip().lower() for l in lineas if len(l.strip()) > 0]))


# Paso 2: Limpiar texto (sin signos ni duplicados)
def limpiar_texto(textos):
    return [re.sub(r"[^a-záéíóúñü\s]", "", t) for t in textos]


# Paso 3: Tokenización
def tokenizar(textos, num_palabras=100):
    tokenizer = TextVectorization(max_tokens=num_palabras)
    tokenizer.adapt(textos)
    return tokenizer


# Paso 4: Extraer vectores con Keras Embedding
def obtener_embeddings(tokenizer, dimension=8):
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential(
        [Embedding(input_dim=vocab_size, output_dim=dimension, input_length=1)]
    )
    palabras = list(tokenizer.word_index.keys())
    indices = np.array([tokenizer.word_index[p] for p in palabras])
    embeddings = model.predict(indices, verbose=0)
    return palabras, embeddings


# Paso 5: Reducción de dimensionalidad con t-SNE
def reducir_con_tsne(vectores):
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=5, init="pca", learning_rate="auto"
    )
    return tsne.fit_transform(vectores)


# Paso 6: Visualización con etiquetas
def graficar_2d(vectores_2d, etiquetas):
    plt.figure(figsize=(10, 8))
    for i, palabra in enumerate(etiquetas):
        x, y = vectores_2d[i]
        plt.scatter(x, y)
        plt.text(x + 0.01, y + 0.01, palabra, fontsize=9)
    plt.title("Visualización 2D de embeddings de palabras")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# --------- MAIN ----------
if __name__ == "__main__":
    try:
        frases = cargar_frases("frases.txt")
        frases_limpias = limpiar_texto(frases)
        tokenizer = tokenizar(frases_limpias, num_palabras=100)
        palabras, vectores = obtener_embeddings(tokenizer, dimension=8)

        assert len(palabras) == vectores.shape[0], (
            "Error: número de palabras no coincide con vectores"
        )

        vectores_2d = reducir_con_tsne(vectores)
        graficar_2d(vectores_2d, palabras)

    except FileNotFoundError:
        print("⚠️ Error: No se encontró el archivo 'frases.txt'")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
