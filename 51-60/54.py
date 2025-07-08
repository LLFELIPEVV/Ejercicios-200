# 🧠 Ejercicio 54/200: Detección de outliers semánticos usando GloVe + similitud coseno
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 1️⃣ Cargar embeddings GloVe desde archivo .txt
def cargar_embeddings(ruta_archivo):
    embedding_index = {}
    with open(ruta_archivo, encoding="utf8") as f:
        for linea in f:
            valores = linea.split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embedding_index[palabra] = vector
    return embedding_index


# 2️⃣ Preparar el conjunto de palabras a analizar
def filtrar_palabras_existentes(palabras, embedding_index):
    return [palabra for palabra in palabras if palabra in embedding_index]


# 3️⃣ Obtener los vectores y calcular similitud con el centro semántico
def detectar_outlier(palabras, embedding_index):
    vectores = np.array([embedding_index[p] for p in palabras])
    centro = np.mean(vectores, axis=0, keepdims=True)  # Vector medio del grupo
    similitudes = cosine_similarity(vectores, centro).flatten()
    idx_outlier = np.argmin(similitudes)
    palabra_outlier = palabras[idx_outlier]
    return similitudes, palabra_outlier


# 📦 Ruta del archivo GloVe
ruta_glove = "Gloove/glove.6B.100d.txt"
embedding_index = cargar_embeddings(ruta_glove)

# 🧾 Lista de palabras del dominio anime + semántica profunda
grupo = [
    # 🧑‍🎤 Personajes
    "goku", "vegeta", "naruto", "sasuke", "luffy", "zoro", "eren", "mikasa", "levi",
    "sakura", "nami", "tanjiro", "nezuko", "saitama", "genos", "itachi",

    # 📚 Cultura anime
    "anime", "manga", "otaku", "cosplay", "waifu", "husbando", "kawaii", "baka", "sensei",
    "senpai", "tsundere", "shonen", "shojo", "seinen", "josei", "isekai", "mecha",

    # 🌀 Poderes y objetos
    "ki", "chakra", "haki", "jutsu", "magic", "spell", "transformation", "power", "sword", "katana",

    # ❤️‍🔥 Emociones y narrativa
    "love", "hate", "friendship", "rivalry", "justice", "evil", "hero", "villain",
    "hope", "despair", "sacrifice", "destiny", "honor", "fear",

    # 🌌 Mundo y contexto
    "universe", "planet", "dimension", "dream", "reality", "future", "past", "present",
    "fire", "water", "wind", "sky", "earth", "light", "darkness",

    # 🧠 Temas de creencias y teoría
    "truth", "evidence", "fact", "rumor", "hoax", "fake", "conspiracy", "science",
    "theory", "belief", "proof", "unbelievable", "mystery", "secret",

    # 🧭 Narrativa y propósito
    "journey", "adventure", "battle", "peace", "war", "destruction", "creation",
    "evolution", "training", "clan", "reincarnation", "betrayal", "legend"
]

# 4️⃣ Filtrar palabras válidas (solo las que están en GloVe)
grupo_filtrado = filtrar_palabras_existentes(grupo, embedding_index)

# 5️⃣ Calcular similitudes y detectar outlier
similitudes, outlier = detectar_outlier(grupo_filtrado, embedding_index)

# 6️⃣ Mostrar resultados
print("🧠 Palabras analizadas:", grupo_filtrado)
print("📊 Similitud con el centro semántico:", np.round(similitudes, 3))
print(f"\n🚨 Palabra outlier detectada: {outlier}")
