# 🧠 Ejercicio 99/200 — Comparativa profesional: Embeddings entrenables vs GloVe (Keras)
# ======================================================================
# Este script compara dos enfoques de representación de texto: embeddings
# entrenables desde cero vs embeddings preentrenados (GloVe), ambos usando
# una red neuronal densa y datos reales de noticias falsas y verdaderas.
# ======================================================================

# =============================
# 📁 Importar librerías necesarias
# =============================
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    TextVectorization,
)
from keras.optimizers import Adam
from keras import backend as K

# ===============================
# 🚀 Optimizar el uso del CPU
# ===============================
num_threads = os.cpu_count()  # Detectamos número de hilos
os.environ["OMP_NUM_THREADS"] = str(num_threads)
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# ==============================
# 📝 Cargar dataset real
# ==============================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()
df_fake["label"] = 0  # Etiqueta: 0 = Fake
df_true["label"] = 1  # Etiqueta: 1 = Real

# Tomamos una muestra mezclada para eficiencia
df = (
    pd.concat([df_fake, df_true])["text"]
    .to_frame()
    .join(pd.concat([df_fake, df_true])["label"])
    .sample(1000, random_state=42)
)

X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================================
# 🌐 TextVectorization: Tokenizar textos
# ======================================
vocab_size = 5000  # Tamaño del vocabulario
max_len = 100  # Longitud máxima de cada secuencia

text_vectorizer = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=max_len,
)

text_vectorizer.adapt(X_train)  # Aprende vocabulario y frecuencias

X_train_vec = text_vectorizer(X_train)
X_test_vec = text_vectorizer(X_test)

# ===================================================
# 🏫 Modelo 1: Embeddings ENTRENABLES desde cero
# ===================================================
model_trainable = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=100),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model_trainable.compile(
    optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

history_trainable = model_trainable.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=32,
    verbose=1,
)

# Predicciones y evaluación
y_pred_trainable = (model_trainable.predict(X_test_vec) > 0.5).astype(int)
print("\n📊 Reporte con embeddings ENTRENADOS desde cero:\n")
print(classification_report(y_test, y_pred_trainable, target_names=["Fake", "Real"]))

# ======================================================
# 🔧 Modelo 2: Embeddings GloVe PREENTRENADOS
# ======================================================
embedding_dim = 100
embedding_index = {}

# Cargamos GloVe (archivo requerido: glove.6B.100d.txt)
with open("Gloove/glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.strip().split()
        word, coefs = values[0], np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# Construimos matriz de embeddings GloVe
word_index = dict(zip(text_vectorizer.get_vocabulary(), range(vocab_size)))
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for word, i in word_index.items():
    if word in embedding_index:
        embedding_matrix[i] = embedding_index[word]

# Creamos modelo con embeddings fijos
model_glove = Sequential(
    [
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,
        ),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model_glove.compile(
    optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

history_glove = model_glove.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=32,
    verbose=1,
)

# Predicciones y evaluación
y_pred_glove = (model_glove.predict(X_test_vec) > 0.5).astype(int)
print("\n📊 Reporte con embeddings GloVe:\n")
print(classification_report(y_test, y_pred_glove, target_names=["Fake", "Real"]))

# ====================================
# 📊 Visualización de resultados con Seaborn
# ====================================
plt.figure(figsize=(10, 5))
sns.lineplot(
    x=range(1, 4),
    y=history_trainable.history["val_accuracy"],
    label="Entrenables",
    marker="o",
)
sns.lineplot(
    x=range(1, 4), y=history_glove.history["val_accuracy"], label="GloVe", marker="s"
)
plt.title("Precisión de validación por época")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 🧹 Limpieza de memoria
K.clear_session()
gc.collect()
