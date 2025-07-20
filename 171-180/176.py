# 🧠 Ejercicio 176/200 — Uso de Embedding Fijo (no entrenable) en Keras con pesos personalizados
import os
import numpy as np

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Flatten, Dense, TextVectorization

# Paso 1: Datos simples de ejemplo (noticias reales y falsas)
textos = [
    "The president visited the disaster area and promised aid",
    "Aliens have landed and taken over the White House",
    "Scientists discover cure for rare disease",
    "Fake news site reports that vaccines cause magnetism",
]
etiquetas = [1, 0, 1, 0]  # 1 = real, 0 = fake

# Paso 2: Tokenizar texto
max_palabras = 1000
max_longitud = 10
tokenizer = TextVectorization(num_words=max_palabras)
tokenizer.adapt(textos)
datos_padded = pad_sequences(tokenizer, maxlen=max_longitud)


# Paso 3: Cargar GloVe y crear embedding matrix
def cargar_glove(filepath, vocabulario, dim=50):
    embeddings = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            if palabra in vocabulario:
                vector = np.array(valores[1:], dtype="float32")
                embeddings[palabra] = vector
    matriz = np.zeros((len(vocabulario) + 1, dim))
    for palabra, idx in vocabulario.items():
        vector = embeddings.get(palabra)
        if vector is not None:
            matriz[idx] = vector
    return matriz


ruta_glove = "glove.6B.50d.txt"
if not os.path.exists(ruta_glove):
    print("❌ Archivo GloVe no encontrado.")
    exit()

embedding_matrix = cargar_glove(ruta_glove, tokenizer.word_index)

# Paso 4: Modelo con capa de embedding NO entrenable
modelo = Sequential()
modelo.add(
    Embedding(
        input_dim=embedding_matrix.shape[0],
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=max_longitud,
        trainable=False,  # 🔒 Importante: No se entrena
    )
)
modelo.add(Flatten())
modelo.add(Dense(1, activation="sigmoid"))

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Paso 5: Entrenar modelo
modelo.fit(datos_padded, np.array(etiquetas), epochs=15, verbose=1)
