# 🧠 Ejercicio 194/200 — Visualización de embeddings con t-SNE para análisis semántico
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from keras.models import Sequential
from keras.layers import Embedding, TextVectorization

# ---------- 1. DEFINIR VOCABULARIO DE PRUEBA ----------
# Lista de palabras que queremos visualizar
palabras = [
    "government",
    "president",
    "virus",
    "cure",
    "doctor",
    "hospital",
    "fake",
    "truth",
    "news",
    "election",
    "pandemic",
    "mask",
    "vaccination",
    "conspiracy",
    "science",
]

# Crear un tokenizer que asigna índices a las palabras
tokenizer = TextVectorization()
tokenizer.adapt(palabras)
word_index = tokenizer.word_index

# Convertimos cada palabra a su índice (para usarlos en la capa Embedding)
indices = [word_index[word] for word in palabras]

# ---------- 2. CREAR CAPA DE EMBEDDING ----------
# Simulamos que estas palabras tienen embeddings preentrenados de 8 dimensiones
model = Sequential()
model.add(Embedding(input_dim=len(word_index) + 1, output_dim=8, input_length=1))
model.compile(optimizer="adam", loss="mse")  # Dummy compile

# Extraer los pesos de la capa de embeddings
embedding_weights = model.layers[0].get_weights()[0]

# Obtener los vectores para nuestras palabras específicas
selected_vectors = embedding_weights[indices]

# ---------- 3. REDUCCIÓN DE DIMENSIONALIDAD CON t-SNE ----------
# Reducir a 2D para visualización
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(selected_vectors)

# ---------- 4. GRAFICAR LOS RESULTADOS ----------
plt.figure(figsize=(10, 6))
for i, palabra in enumerate(palabras):
    x, y = embeddings_2d[i]
    plt.scatter(x, y)
    plt.text(x + 0.01, y + 0.01, palabra, fontsize=9)

plt.title("Visualización 2D de embeddings con t-SNE")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_embeddings_fake_news.png")
plt.show()
