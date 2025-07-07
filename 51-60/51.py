# 🧠 Ejercicio 51/200: Visualización de embeddings GloVe usando t-SNE (análisis semántico)
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

# 1️⃣ Cargar vectores preentrenados GloVe (100 dimensiones por palabra)
embedding_index = {}
ruta_glove = "Gloove/glove.6B.100d.txt"

with open(ruta_glove, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]  # palabra como string
        coefs = np.asarray(values[1:], dtype="float32")  # vector GloVe
        embedding_index[word] = coefs

# 2️⃣ Definir subconjunto de palabras clave relacionadas con noticias y fake news
selected_words = [
    "truth",
    "fact",
    "real",
    "fake",
    "lie",
    "fraud",
    "trust",
    "news",
    "government",
    "media",
    "science",
    "hoax",
    "evidence",
    "conspiracy",
    "rumor",
    "false",
    "accurate",
    "bias",
    "clickbait",
    "social",
    "platform",
]

# 3️⃣ Filtrar solo las palabras que existen en GloVe (evita errores)
selected_words = [word for word in selected_words if word in embedding_index]

# 4️⃣ Obtener los vectores de las palabras seleccionadas
word_vectors = np.array([embedding_index[word] for word in selected_words])

# 5️⃣ Reducir dimensionalidad con t-SNE (de 100D a 2D para visualización)
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
word_vectors_2d = tsne.fit_transform(word_vectors)

# 6️⃣ Visualizar en un gráfico 2D con etiquetas
plt.figure(figsize=(12, 8))
for i, word in enumerate(selected_words):
    x, y = word_vectors_2d[i]
    plt.scatter(x, y, color="blue")
    plt.text(x + 0.1, y + 0.1, word, fontsize=10, color="black")

plt.title("Visualización de embeddings GloVe con t-SNE (Fake News Context)")
plt.grid(True)
plt.tight_layout()
plt.show()
