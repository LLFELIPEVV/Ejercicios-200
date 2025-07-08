# 🧠 Ejercicio 52/200: Análisis semántico con GloVe y visualización de similitud coseno
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# 1️⃣ Cargar embeddings preentrenados GloVe (100 dimensiones)
embedding_index = {}

# 💡 Se recomienda encapsular esto en una función para mejor reutilización
with open("Gloove/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]  # Primera palabra es el token
        vector = np.asarray(values[1:], dtype="float32")  # El resto es el vector
        embedding_index[word] = vector

# 2️⃣ Lista de palabras relevantes para el análisis semántico
words = [
    # 🧑‍🎤 Personajes populares (shonen & mainstream)
    "goku", "vegeta", "naruto", "sasuke", "luffy", "zoro", "eren", "mikasa", "levi",
    "sakura", "nami", "tanjiro", "nezuko", "saitama", "genos", "itachi",

    # 📚 Términos del mundo anime/manga
    "anime", "manga", "otaku", "cosplay", "waifu", "husbando", "kawaii", "baka", "sensei",
    "senpai", "tsundere", "dere", "shonen", "shojo", "seinen", "josei", "isekai", "mecha",

    # 🌀 Poderes y habilidades comunes
    "ki", "chakra", "haki", "jutsu", "magic", "spell", "transformation", "power", "strength",
    "speed", "agility", "teleportation", "sword", "katana", "energy",

    # 🧠 Emociones y conceptos típicos
    "love", "hate", "friendship", "rivalry", "justice", "evil", "hero", "villain", "honor",
    "courage", "despair", "hope", "destiny", "fear", "sacrifice",

    # 🌍 Escenarios y mundos
    "world", "universe", "dimension", "planet", "reality", "dream", "nightmare", "future",
    "past", "present", "earth", "sky", "sea", "fire", "water", "wind", "light", "darkness",

    # 🔮 Temas frecuentes en teorías de fans
    "truth", "evidence", "fact", "rumor", "hoax", "fake", "conspiracy", "science", "proof",
    "theory", "believe", "unbelievable", "mystery", "secret",

    # 🧭 Conceptos narrativos comunes
    "journey", "adventure", "battle", "war", "peace", "destruction", "creation",
    "evolution", "reincarnation", "betrayal", "training", "clan", "curse", "legend"
]

# 3️⃣ Filtrar solo las palabras que están en el vocabulario de GloVe
words = [word for word in words if word in embedding_index]

# 4️⃣ Seleccionar 20 palabras aleatorias para visualizar (🧪 semilla para reproducibilidad)
np.random.seed(42)
words_sample = np.random.choice(words, size=20, replace=False)


# 5️⃣ Función para calcular la matriz de similitud coseno
def get_similarity_matrix(words, embeddings):
    vectors = np.array([embeddings[word] for word in words])
    return cosine_similarity(vectors)


# 6️⃣ Generar matriz de similitud
similarity_matrix = get_similarity_matrix(words_sample, embedding_index)

# 7️⃣ Visualización con heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    xticklabels=words_sample,
    yticklabels=words_sample,
    cmap="YlGnBu",
    annot=True,
    fmt=".2f",
)
plt.title("Similitud Coseno entre Palabras (GloVe)")
plt.xlabel("Palabras")
plt.ylabel("Palabras")
plt.tight_layout()
plt.show()
