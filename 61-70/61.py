# ✅ Ejercicio 61/200 — Comparación de GloVe promedio vs. TextVectorization en un clasificador binario profesional
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
    Input,
    TextVectorization,
)
from keras.optimizers import Adam

# 🧩 1. Configuración general reproducible
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 📥 2. Carga de datos reales de noticias
fake = pd.read_csv("Datasets/archive/Fake.csv")
true = pd.read_csv("Datasets/archive/True.csv")

fake["label"] = 0
true["label"] = 1

# 📊 Concatenación y limpieza básica
df = pd.concat([fake, true], ignore_index=True)[["text", "label"]].dropna()
X = df["text"]
y = df["label"]

# 🔀 3. División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


# 🧠 4A. Carga de embeddings preentrenados GloVe
def cargar_embeddings(path):
    embeddings = {}
    with open(path, encoding="utf8") as f:
        for linea in f:
            valores = linea.strip().split()
            palabra = valores[0]
            vector = np.asarray(valores[1:], dtype="float32")
            embeddings[palabra] = vector
    return embeddings


def texto_a_vector(texto, embeddings, dim=100):
    palabras = texto.lower().split()
    vectores = [embeddings[p] for p in palabras if p in embeddings]
    return np.mean(vectores, axis=0) if vectores else np.zeros(dim)


glove_path = "Gloove/glove.6B.100d.txt"
embedding_index = cargar_embeddings(glove_path)

# 🔁 Vectorización por promedio de GloVe
X_train_glove = np.array([texto_a_vector(t, embedding_index) for t in X_train])
X_test_glove = np.array([texto_a_vector(t, embedding_index) for t in X_test])


# 🏗 5A. Modelo simple con GloVe (input vector de 100 dimensiones)
def crear_modelo_glove():
    model = Sequential(
        [
            Input(shape=(100,)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


model_glove = crear_modelo_glove()
model_glove.fit(
    X_train_glove, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# 📈 Evaluación modelo GloVe
y_pred_g = (model_glove.predict(X_test_glove).flatten() > 0.5).astype(int)
print("\n📊 GloVe — Reporte de clasificación:")
print(classification_report(y_test, y_pred_g, zero_division=0))

# 🧠 4B. TextVectorization + Embedding entrenable
vectorizador = TextVectorization(
    max_tokens=10000, output_mode="int", output_sequence_length=300
)
vectorizador.adapt(X_train.values)

X_train_seq = vectorizador(X_train)
X_test_seq = vectorizador(X_test)


# 🏗 5B. Modelo con TextVectorization + Embedding + Pooling
def crear_modelo_textvec():
    model = Sequential(
        [
            Input(shape=(300,)),
            Embedding(input_dim=10000, output_dim=100),  # Vector entrenable
            GlobalAveragePooling1D(),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


model_vec = crear_modelo_textvec()
model_vec.fit(
    X_train_seq, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1
)

# 📈 Evaluación modelo con TextVectorization
y_pred_v = (model_vec.predict(X_test_seq).flatten() > 0.5).astype(int)
print("\n📊 TextVectorization — Reporte de clasificación:")
print(classification_report(y_test, y_pred_v, zero_division=0))
