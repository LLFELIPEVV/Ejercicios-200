# 🧠 Ejercicio 179/200 — Comparar LSTM vs. modelo clásico usando embeddings congelados
import time
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, TextVectorization

# Paso 1: Cargar dataset simple
df = pd.read_csv("fake_or_real_news.csv")
textos = df["text"].astype(str).values
etiquetas = df["label"].values

# Convertir etiquetas 'REAL'/'FAKE' a 0/1
le = LabelEncoder()
y = le.fit_transform(etiquetas)

# Paso 2: Tokenizar texto
MAX_PALABRAS = 10000
MAX_LONGITUD = 100

tokenizer = TextVectorization(max_tokens=MAX_PALABRAS)
tokenizer.adapt(textos)
x = pad_sequences(tokenizer, maxlen=MAX_LONGITUD, padding="post")

# Paso 3: Dividir conjunto de datos
x_entrenamiento, x_test, y_entrenamiento, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# Paso 4: Cargar embeddings GloVe 50d
def cargar_glove(ruta, dimension, tokenizer):
    embedding_index = {}
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            valores = linea.split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embedding_index[palabra] = vector

    vocabulario = tokenizer.word_index
    matriz = np.zeros((MAX_PALABRAS, dimension))
    for palabra, i in vocabulario.items():
        if i < MAX_PALABRAS:
            vector = embedding_index.get(palabra)
            if vector is not None:
                matriz[i] = vector
    return matriz


ruta_glove = "glove.6B.50d.txt"
embedding_matrix = cargar_glove(ruta_glove, 50, tokenizer)


# Paso 5: Clasificador clásico usando promedio de embeddings
def promedio_embeddings(x, embedding_matrix):
    embeddings = []
    for secuencia in x:
        vectores = []
        for idx in secuencia:
            if idx < len(embedding_matrix):
                vectores.append(embedding_matrix[idx])
        if vectores:
            embeddings.append(np.mean(vectores, axis=0))
        else:
            embeddings.append(np.zeros(embedding_matrix.shape[1]))
    return np.array(embeddings)


print("\n🔎 Evaluando LogisticRegression...")
start = time.time()
x_train_avg = promedio_embeddings(x_entrenamiento, embedding_matrix)
x_test_avg = promedio_embeddings(x_test, embedding_matrix)

modelo_lr = LogisticRegression(max_iter=500)
modelo_lr.fit(x_train_avg, y_entrenamiento)
y_pred_lr = modelo_lr.predict(x_test_avg)
acc_lr = accuracy_score(y_test, y_pred_lr)
end = time.time()
print(f"✅ LogisticRegression Accuracy: {acc_lr:.4f} | Tiempo: {end - start:.2f} seg")

# Paso 6: LSTM mínima con embeddings congelados
print("\n🔎 Evaluando LSTM (embeddings congelados)...")
start = time.time()
modelo_lstm = Sequential(
    [
        Embedding(
            MAX_PALABRAS,
            50,
            weights=[embedding_matrix],
            input_length=MAX_LONGITUD,
            trainable=False,
        ),
        LSTM(32),
        Dense(1, activation="sigmoid"),
    ]
)

modelo_lstm.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
modelo_lstm.fit(x_entrenamiento, y_entrenamiento, epochs=5, batch_size=32, verbose=0)
loss, acc_lstm = modelo_lstm.evaluate(x_test, y_test, verbose=0)
end = time.time()
print(f"✅ LSTM Accuracy: {acc_lstm:.4f} | Tiempo: {end - start:.2f} seg")
