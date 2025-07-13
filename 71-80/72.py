# ✅ Ejercicio 72/200 — Comparación entre Embedding desde cero vs. GloVe preentrenado
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    LSTM,
    Dense,
)
from keras.optimizers import Adam

# -----------------------------
# 📥 1. Carga y limpieza de datos
# -----------------------------

df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# -----------------------------
# 🔠 2. Vectorización del texto
# -----------------------------

vectorizer = TextVectorization(
    max_tokens=10000, output_sequence_length=300, output_mode="int"
)
vectorizer.adapt(X_train)

X_train_seq = vectorizer(X_train)
X_test_seq = vectorizer(X_test)

# -----------------------------
# 📦 3. Cargar embeddings GloVe
# -----------------------------


def cargar_embeddings(path, dim=100):
    """Carga archivo GloVe y devuelve diccionario palabra → vector"""
    embeddings_index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            valores = line.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings_index[palabra] = vector
    return embeddings_index


glove_index = cargar_embeddings("Gloove/glove.6B.100d.txt", dim=100)

# Obtener vocabulario del vectorizador
vocab = vectorizer.get_vocabulary()
word_index = dict(zip(vocab, range(len(vocab))))

# Crear matriz de embeddings para inicializar capa
embedding_matrix = np.zeros((len(vocab), 100))
for palabra, idx in word_index.items():
    vector = glove_index.get(palabra)
    if vector is not None:
        embedding_matrix[idx] = vector

# -----------------------------
# 🧠 4. Modelo 1: Embedding desde cero
# -----------------------------

model_random = Sequential(
    [
        Input(shape=(300,)),
        Embedding(input_dim=10000, output_dim=100),
        LSTM(64),
        Dense(1, activation="sigmoid"),
    ]
)
model_random.compile(
    optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

# -----------------------------
# 🧠 5. Modelo 2: Embedding con GloVe
# -----------------------------

model_glove = Sequential(
    [
        Input(shape=(300,)),
        Embedding(
            input_dim=len(vocab),
            output_dim=100,
            weights=[embedding_matrix],
            trainable=False,  # ❄️ No se entrena (fijo)
        ),
        LSTM(64),
        Dense(1, activation="sigmoid"),
    ]
)
model_glove.compile(
    optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

# -----------------------------
# 🏋️ 6. Entrenamiento de ambos modelos
# -----------------------------

print("🔧 Entrenando modelo con Embedding desde cero...")
model_random.fit(X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1)

print("🔧 Entrenando modelo con GloVe preentrenado...")
model_glove.fit(X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1)

# -----------------------------
# 📊 7. Evaluación y comparación
# -----------------------------

print("\n📈 Clasificación - Embedding desde cero:")
y_pred = (model_random.predict(X_test_seq).flatten() > 0.5).astype(int)
print(classification_report(y_test, y_pred, zero_division=0))

print("\n📈 Clasificación - GloVe Embedding:")
y_pred = (model_glove.predict(X_test_seq).flatten() > 0.5).astype(int)
print(classification_report(y_test, y_pred, zero_division=0))
