# 🧪 Ejercicio 191/200 — Visualización de Embeddings con PCA y t-SNE
# coding: utf-8
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from keras.models import Sequential
from keras.layers import Embedding, TextVectorization
from keras.preprocessing.sequence import pad_sequences

# 1. Corpus reducido de entrenamiento
corpus = [
    "government confirms economy growth",  # real
    "scientists discover water on mars",  # real
    "breaking aliens stole the moon",  # fake
    "you won't believe this miracle cure",  # fake
    "NASA announces new space mission",  # real
    "doctors shocked by this discovery",  # fake
]

# 2. Tokenización (asignar número a cada palabra)
tokenizer = TextVectorization(max_tokens=1000)
tokenizer.adapt(corpus)
word_index = tokenizer.word_index
print(f"🔠 Palabras encontradas: {list(word_index.items())[:8]}")

# Convertir los textos en secuencias numéricas
sequences = tokenizer.texts_to_sequences(corpus)
padded = pad_sequences(sequences, maxlen=6, padding="post")  # fijo a 6 palabras

# 3. Modelo solo con Embedding (no entrenamiento adicional)
vocab_size = len(word_index) + 1  # +1 porque el index comienza en 1
embed_dim = 8

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=6))
model.compile("adam", "mse")

# Forzar una pasada para que se inicialicen los pesos
model.predict(padded)

# 4. Extraer la matriz de embeddings (vector por palabra)
embedding_layer = model.layers[0]
embedding_weights = embedding_layer.get_weights()[0]  # shape: (vocab_size, embed_dim)

# 5. Mapear palabras a sus vectores
words = list(word_index.keys())
vectors = embedding_weights[1 : len(words) + 1]  # ignorar token 0

# 6. Reducir dimensiones: PCA y luego t-SNE para mejor separación visual
pca = PCA(n_components=16)  # Primero bajamos de 8D a 16D
pca_result = pca.fit_transform(vectors)

tsne = TSNE(n_components=2, perplexity=5, n_iter=1000, random_state=42)
tsne_result = tsne.fit_transform(pca_result)

# 7. Visualizar en 2D
plt.figure(figsize=(10, 7))
for i, word in enumerate(words):
    x, y = tsne_result[i]
    plt.scatter(x, y)
    plt.annotate(word, (x + 0.2, y + 0.2), fontsize=9)

plt.title("🧠 Visualización de Embeddings con t-SNE")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("embedding_visualizacion.png")
plt.show()
