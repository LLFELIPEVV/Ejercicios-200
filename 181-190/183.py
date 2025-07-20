# 🎯 Ejercicio 183/200 — Pipeline completo de inferencia con embeddings GloVe preentrenados
import os
import re
import html
import numpy as np

from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, TextVectorization


# Paso 1: Sanitización básica del texto
def sanitizar(texto):
    texto = html.unescape(texto)
    texto = re.sub(r"<[^>]+>", "", texto)
    texto = re.sub(r"[^a-zA-Z\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    texto = texto.lower()
    return texto


# Paso 2: Cargar los vectores GloVe preentrenados (50 dimensiones)
def cargar_glove(filepath, vocab_tokenizer, dimension=50):
    print("Cargando vectores GloVe...")
    embeddings_index = {}
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            valores = line.split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings_index[palabra] = vector

    print("Construyendo matriz de embeddings...")
    vocab_size = len(vocab_tokenizer.word_index) + 1
    matriz_embeddings = np.zeros((vocab_size, dimension))

    for palabra, idx in vocab_tokenizer.word_index.items():
        vector = embeddings_index.get(palabra)
        if vector is not None:
            matriz_embeddings[idx] = vector
        # si no está en GloVe, se deja como vector de ceros

    return matriz_embeddings


# Paso 3: Crear modelo dummy con embeddings fijos (no entrenable)
def construir_modelo(vocab_size, embedding_matrix, input_length):
    model = Sequential()
    model.add(
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_matrix.shape[1],
            weights=[embedding_matrix],
            input_length=input_length,
            trainable=False,
        )
    )
    model.add(LSTM(32))  # simple, rápido
    model.add(Dense(1, activation="sigmoid"))  # salida binaria: fake o real
    model.compile(optimizer="adam", loss="binary_crossentropy")
    return model


# Paso 4: Inferencia (modelo no entrenado aún)
def inferir(modelo, texto, tokenizer, max_len):
    texto_limpio = sanitizar(texto)
    secuencia = tokenizer.texts_to_sequences([texto_limpio])
    entrada = pad_sequences(secuencia, maxlen=max_len)
    pred = modelo.predict(entrada, verbose=0)
    return f"🧠 Predicción (simulada): {'Fake' if pred[0][0] > 0.5 else 'Real'} (prob={pred[0][0]:.2f})"


# Paso 5: Inicialización del pipeline
if __name__ == "__main__":
    print("=== Iniciando pipeline de inferencia con GloVe ===\n")

    # Corpus base para ajustar el tokenizer (como si fuera el corpus real)
    corpus = [
        "Vaccines are a hoax according to some people",
        "The government officially denied the report",
        "Win a million dollars now click here",
        "Breaking news: scientists discover new particle",
    ]
    tokenizer = TextVectorization(max_tokens=1000)
    tokenizer.adapt([sanitizar(t) for t in corpus])

    # Carga de GloVe (se requiere archivo 'glove.6B.50d.txt')
    ruta_glove = "glove.6B.50d.txt"
    if not os.path.exists(ruta_glove):
        print("❌ ERROR: Archivo de GloVe no encontrado. Descárgalo desde:")
        print("   https://nlp.stanford.edu/data/glove.6B.zip (usa glove.6B.50d.txt)")
        exit()

    matriz_emb = cargar_glove(ruta_glove, tokenizer)
    modelo = construir_modelo(
        vocab_size=matriz_emb.shape[0], embedding_matrix=matriz_emb, input_length=30
    )

    entrada = input("\nIntroduce una noticia: ")
    resultado = inferir(modelo, entrada, tokenizer, max_len=30)
    print(resultado)
