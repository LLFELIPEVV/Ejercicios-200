# 🛡️ Ejercicio 180/200 — Inferencia segura ante entradas maliciosas simples
import re
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.layers import TextVectorization
from keras.preprocessing.sequence import pad_sequences

# Paso 1: Cargar datos y preparar embeddings
df = pd.read_csv("fake_or_real_news.csv")
textos = df["text"].astype(str).values
etiquetas = df["label"].values

le = LabelEncoder()
y = le.fit_transform(etiquetas)

# Tokenización
MAX_PALABRAS = 10000
MAX_LONGITUD = 100

tokenizer = TextVectorization(max_tokens=MAX_PALABRAS)
tokenizer.adapt(textos)
x = pad_sequences(tokenizer, maxlen=MAX_LONGITUD, padding="post")


# Embeddings GloVe congelados
def cargar_glove(ruta, dimension, tokenizer):
    embedding_index = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            valores = linea.split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embedding_index[palabra] = vector
    matriz = np.zeros((MAX_PALABRAS, dimension))
    vocab = tokenizer.word_index
    for palabra, i in vocab.items():
        if i < MAX_PALABRAS and palabra in embedding_index:
            matriz[i] = embedding_index[palabra]
    return matriz


embedding_matrix = cargar_glove("glove.6B.50d.txt", 50, tokenizer)


# Promedio de embeddings por texto
def promedio_embeddings(x, matriz):
    salida = []
    for secuencia in x:
        vectores = [matriz[i] for i in secuencia if i < len(matriz)]
        if vectores:
            salida.append(np.mean(vectores, axis=0))
        else:
            salida.append(np.zeros(matriz.shape[1]))
    return np.array(salida)


# Modelo simple para prueba
x_avg = promedio_embeddings(x, embedding_matrix)
x_train, x_test, y_train, y_test = train_test_split(
    x_avg, y, test_size=0.2, random_state=42
)
modelo = LogisticRegression(max_iter=500)
modelo.fit(x_train, y_train)


# Paso 2: Funciones de limpieza y detección de entradas maliciosas
def sanitizar(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)  # eliminar URLs
    texto = re.sub(r"[^a-zA-Z\s]", "", texto)  # quitar símbolos
    texto = re.sub(r"\s+", " ", texto).strip()  # quitar espacios múltiples
    return texto


def es_malicioso(texto):
    # Patrón 1: Palabras repetidas más de 3 veces seguidas
    if re.search(r"(\b\w+\b)(\s+\1){3,}", texto):
        return True
    # Patrón 2: Demasiadas mayúsculas (antes de limpieza)
    if sum(1 for c in texto if c.isupper()) > len(texto) * 0.5:
        return True
    # Patrón 3: Texto extremadamente corto o largo
    if len(texto.split()) < 3 or len(texto.split()) > 300:
        return True
    return False


# Paso 3: Función de inferencia segura
def inferir_noticia(texto):
    print("\n📰 Entrada original:", texto[:100], "..." if len(texto) > 100 else "")
    if es_malicioso(texto):
        print("🚫 Entrada rechazada: posible intento malicioso o inconsistente.")
        return
    limpio = sanitizar(texto)
    secuencia = tokenizer.texts_to_sequences([limpio])
    padded = pad_sequences(secuencia, maxlen=MAX_LONGITUD)
    promedio = promedio_embeddings(padded, embedding_matrix)
    pred = modelo.predict(promedio)[0]
    etiqueta = le.inverse_transform([pred])[0]
    print("✅ Predicción:", etiqueta)


# Paso 4: Ejemplos de inferencia
ejemplos = [
    "Breaking: New study shows vaccination reduces spread of virus!",
    "FAKE FAKE FAKE FAKE FAKE FAKE FAKE FAKE",
    "THEY LIED TO YOU AGAIN THIS IS NOT NEWS!!!",
    "The economy is recovering slowly, say experts in the field.",
]

for texto in ejemplos:
    inferir_noticia(texto)
