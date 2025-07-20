# 🧪 Ejercicio 188/200 — Visualización de Embeddings con PCA y t-SNE
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization

# 1. Corpus reducido
corpus = [
    "breaking news president signs law",
    "aliens land in new york",
    "government confirms growth",
    "celebrity cloned by agency",
    "scientists find vaccine",
    "hoax news goes viral",
    "real news government updates",
    "false claims on tv",
    "vaccine stops disease",
    "economy stable this year",
]
etiquetas = [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]  # 1=real, 0=fake

# 2. Tokenización
tokenizer = TextVectorization(max_tokens=1000)
tokenizer.adapt(corpus)
X = pad_sequences(tokenizer, maxlen=6, padding="post")
y = np.array(etiquetas)

# 3. Modelo con embedding entrenado desde cero
modelo = Sequential(
    [
        Embedding(input_dim=1000, output_dim=8, input_length=6),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]
)
modelo.compile(optimizer="adam", loss="binary_crossentropy")
modelo.fit(X, y, epochs=50, verbose=0)

# 4. Extraer pesos del embedding
embedding_layer = modelo.layers[0]
pesos = embedding_layer.get_weights()[0]

# 5. Palabras a visualizar (primeros 20 tokens del tokenizer)
palabras = list(tokenizer.word_index.keys())[:20]
indices = [tokenizer.word_index[p] for p in palabras if tokenizer.word_index[p] < 1000]
vectores = pesos[indices]

# 6. PCA
pca = PCA(n_components=2)
vectores_pca = pca.fit_transform(vectores)

# 7. t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
vectores_tsne = tsne.fit_transform(vectores)


# 8. Función para graficar
def graficar(vectores_2d, titulo):
    plt.figure(figsize=(8, 5))
    for i, palabra in enumerate(palabras):
        if tokenizer.word_index[palabra] < 1000:
            x, y = vectores_2d[i]
            plt.scatter(x, y)
            plt.text(x + 0.01, y + 0.01, palabra)
    plt.title(titulo)
    plt.grid(True)
    plt.show()


# 9. Visualizaciones
graficar(vectores_pca, "🔍 Embeddings con PCA")
graficar(vectores_tsne, "🔍 Embeddings con t-SNE")
