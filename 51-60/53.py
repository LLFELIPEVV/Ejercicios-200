# 🧠 Ejercicio 53/200: Clustering semántico de palabras anime y fake news usando GloVe + KMeans
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

embedding_index = {}
with open("Gloove/glove.6B.100d.txt", encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = vector

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

words = [w for w in words if w in embedding_index]
vectors = np.array([embedding_index[w] for w in words])

num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
labels = kmeans.fit_predict(vectors)

pca = PCA(n_components=2, random_state=42)
vectors_2d = pca.fit_transform(vectors)

plt.figure(figsize=(10, 7))
for i, word in enumerate(words):
    x, y = vectors_2d[i]
    plt.scatter(x, y, c=f"C{labels[i]}", label=f"Cluster {labels[i]}", s=50)
    plt.text(x + 0.01, y + 0.01, word, fontsize=9)
plt.title("Clustering semántico con GloVe + KMeans")
plt.grid(True)
plt.tight_layout()
plt.show()
