# 🧠 Ejercicio 55/200: Detección de frases anómalas usando GloVe + Similitud Coseno + Promedio de Embeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 1️⃣ Función para cargar los embeddings GloVe en un diccionario
def cargar_embeddings(ruta_glove):
    """
    Carga los vectores GloVe desde un archivo .txt y los devuelve en un diccionario.
    Cada clave es una palabra, y su valor es un vector de 100 dimensiones.
    """
    embedding_index = {}
    with open(ruta_glove, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embedding_index[palabra] = vector
    return embedding_index

# 2️⃣ Función para convertir una frase a un vector promedio de embeddings
def frase_a_vector(frase, embedding_index):
    """
    Convierte una frase a un vector usando el promedio de los embeddings
    de las palabras contenidas en la frase (si están en el vocabulario GloVe).
    """
    palabras = frase.lower().split()
    vectores = [embedding_index[p] for p in palabras if p in embedding_index]
    return np.mean(vectores, axis=0) if vectores else None

# 3️⃣ Frases icónicas del anime (puedes incluir frases reales y absurdas para detectar outliers)
frases = [
    "It's over 9000!",
    "Believe it!",
    "I'll take a potato chip... AND EAT IT!",
    "My drill is the drill that will pierce the heavens!",
    "The world isn't perfect, but it's there for us, doing the best it can.",
    "If you don't take risks, you can't create a future!",
    "People die when they are killed.",
    "A lesson without pain is meaningless.",
    "There's no such thing as a painless lesson, they just don't exist.",
    "Hard work is worthless for those that don't believe in themselves.",
    "Bang.",
    "Dattebayo!",
    "I am the bone of my sword.",
    "Madara Uchiha, the strongest.",
    "Pikachu, I choose you!",
    "To know sorrow is not evil. What is evil is to forget it.",
    "I'm just a guy who's a hero for fun.",
    "If you want to be strong, stop caring about what others think of you.",
    "Yare yare daze...",
    "For the sake of our beautiful world!",
]

# 4️⃣ Cargar embeddings GloVe (100 dimensiones)
glove_path = "Gloove/glove.6B.100d.txt"
embedding_index = cargar_embeddings(glove_path)

# 5️⃣ Procesar frases válidas (las que tienen al menos una palabra presente en GloVe)
vectores_frases = []
frases_validas = []

for frase in frases:
    vector = frase_a_vector(frase, embedding_index)
    if vector is not None:
        vectores_frases.append(vector)
        frases_validas.append(frase)

# 6️⃣ Cálculo del centro semántico y similitudes coseno
vectores = np.array(vectores_frases)
centro = np.mean(vectores, axis=0, keepdims=True)
similitudes = cosine_similarity(vectores, centro).flatten()

# 7️⃣ Detección de outlier (la frase con menor similitud al centro)
indice_outlier = np.argmin(similitudes)
frase_outlier = frases_validas[indice_outlier]

# 8️⃣ Reporte final
print("📋 Frases analizadas:")
for frase in frases_validas:
    print("•", frase)

print("\n📉 Similitud coseno con el centro semántico:")
for f, sim in zip(frases_validas, similitudes):
    print(f"→ {sim:.4f} :: {f}")

print(f"\n🚨 Frase outlier detectada (menos relacionada con las demás):\n👉 \"{frase_outlier}\"")
