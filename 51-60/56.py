# 🧠 Ejercicio 56/200: Detección de frases incoherentes en un párrafo usando GloVe + Similitud Coseno
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# 1️⃣ Carga los embeddings GloVe desde archivo
def cargar_embeddings(path):
    index = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            index[palabra] = vector
    return index


# 2️⃣ Convierte una frase a un vector promedio de embeddings
def frase_a_vector(frase, embeddings):
    palabras = frase.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    return np.mean(vectores, axis=0) if vectores else None


# 3️⃣ Función para detectar la frase outlier en un párrafo
def detectar_frase_incoherente(parrafo, embeddings):
    vectores, frases_validas = [], []

    for frase in parrafo:
        vector = frase_a_vector(frase, embeddings)
        if vector is not None:
            vectores.append(vector)
            frases_validas.append(frase)

    if len(vectores) < 2:
        raise ValueError("Se necesitan al menos 2 frases válidas para comparar.")

    # Calcula el centro semántico del párrafo
    vectores = np.array(vectores)
    centro = np.mean(vectores, axis=0, keepdims=True)
    similitudes = cosine_similarity(vectores, centro).flatten()

    # Frase más lejana al centro
    indice_outlier = np.argmin(similitudes)
    frase_incoherente = frases_validas[indice_outlier]

    return frases_validas, similitudes, frase_incoherente


# 4️⃣ Párrafo con frases humorísticas mezcladas
parrafo = [
    "Goku entrenó incansablemente para superar sus límites. Sus batallas épicas definieron el destino del universo. Sin embargo, siempre olvidaba dónde dejó sus llaves del coche.",
    "Naruto soñaba con ser Hokage y nunca se rendía. Completó misiones peligrosas con éxito. Un día, decidió que prefería ser un panadero en Konoha.",
    "Luffy reunió una tripulación diversa para encontrar el One Piece. Su goma le permitía estirarse. A veces, por las noches, le gustaba tejer bufandas para todos.",
    "Eren Jaeger juró eliminar a todos los Titanes. Luchó con determinación inquebrantable. Por las tardes, se dedicaba a coleccionar sellos.",
    "Saitama era el héroe más fuerte. Ayudaba a la gente por diversión. Su mayor debilidad era no poder abrir un frasco de pepinillos.",
    "Sakura Haruno fue una poderosa ninja médica. Fue fundamental en misiones importantes. En secreto, dedicaba sus tardes a escribir fanfiction sobre su sensei.",
    "Vegeta siempre buscó superar a Goku. Se volvió un aliado crucial de la Tierra. A menudo, se le veía en el jardín hablando con plantas.",
    "Mikasa era una guerrera formidable con una lealtad inquebrantable. Protegía a sus seres queridos. No podía resistirse a ver telenovelas dramáticas.",
]

# 5️⃣ Ejecutar el análisis
glove_path = "Gloove/glove.6B.100d.txt"
embedding_index = cargar_embeddings(glove_path)

frases, similitudes, outlier = detectar_frase_incoherente(parrafo, embedding_index)

# 6️⃣ Mostrar resultados
print("📋 Frases analizadas:")
for frase in frases:
    print("•", frase)

print("\n📉 Similitud con centro semántico:")
for frase, sim in zip(frases, similitudes):
    print(f"→ {sim:.4f} :: {frase}")

print(f"\n🚨 Frase incoherente detectada:\n👉 {outlier}")
