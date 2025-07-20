# 🔎 Ejercicio 197/200 — Visualización de Embeddings con PCA y t-SNE
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from gensim.models import KeyedVectors

# ------------------ Cargar Embedding Word2Vec (formato texto) ------------------ #
# Archivo debe estar en formato Word2Vec, si es GloVe original debes convertirlo antes
ruta_modelo = "glove.6B.100d.word2vec.txt"  # O el modelo que tengas disponible

# Cargar sólo si tienes suficiente RAM (puedes usar una versión pequeña)
print("Cargando modelo...")
modelo = KeyedVectors.load_word2vec_format(ruta_modelo, binary=False)

# ------------------ Palabras a visualizar ------------------ #
palabras_clave = [
    "fake",
    "hoax",
    "fraud",
    "rumor",
    "scam",  # fake news
    "true",
    "real",
    "evidence",
    "verified",
    "fact",  # real news
    "virus",
    "vaccine",
    "government",
    "science",  # contexto neutral
]

# Verifica que existan en el modelo (algunos embeddings no tienen todas)
palabras = [p for p in palabras_clave if p in modelo]

# Extraer vectores
vectores = [modelo[p] for p in palabras]

# ------------------ Reducción con PCA ------------------ #
pca = PCA(n_components=2)
vectores_pca = pca.fit_transform(vectores)

# ------------------ Reducción con t-SNE ------------------ #
tsne = TSNE(n_components=2, perplexity=5, init="random", random_state=42)
vectores_tsne = tsne.fit_transform(vectores)


# ------------------ Visualización ------------------ #
def graficar(vectores_2d, titulo):
    plt.figure(figsize=(8, 6))
    for i, palabra in enumerate(palabras):
        x, y = vectores_2d[i]
        plt.scatter(x, y, color="blue")
        plt.annotate(palabra, (x + 0.01, y + 0.01))
    plt.title(titulo)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Mostrar ambas visualizaciones
graficar(vectores_pca, "Visualización PCA de Embeddings")
graficar(vectores_tsne, "Visualización t-SNE de Embeddings")
