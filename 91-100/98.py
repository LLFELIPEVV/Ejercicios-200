# 🧠 Ejercicio 98/200 — Comparativa profesional: TextVectorization vs GloVe embeddings en red densa de Keras

# ----------------------
# 1. Importación de librerías necesarias
# ----------------------
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    TextVectorization,
    Dense,
    Input,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
)

# ----------------------
# 2. Configuración de hilos del procesador
# Esto permite que TensorFlow use eficientemente tu CPU
# ----------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32  # Tamaño de lote pequeño para Ryzen 3 2200U

# ----------------------
# 3. Carga y preparación de datos (Fake vs Real News)
# ----------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()
df_fake["label"] = 0  # Noticias falsas = 0
df_true["label"] = 1  # Noticias reales = 1

# Mezclamos y tomamos una muestra pequeña para evitar sobrecarga
df = pd.concat([df_fake, df_true])[["text", "label"]].sample(1000, random_state=42)
X, y = df["text"].values, df["label"].values

# Dividimos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ====================================================
# 🔹 A. MODELO CON TEXTVECTORIZATION (TF-IDF)
# ====================================================

# Convertimos texto en vectores numéricos usando Keras
vectorizer = TextVectorization(max_tokens=5000, output_mode="tf_idf")
vectorizer.adapt(
    tf.convert_to_tensor(X_train)
)  # Aprende el vocabulario de entrenamiento

# Transformamos los textos a vectores
X_train_vec = vectorizer(tf.convert_to_tensor(X_train)).numpy()
X_test_vec = vectorizer(tf.convert_to_tensor(X_test)).numpy()


# Definimos una red neuronal simple
def build_dense_model(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(32, activation="relu"),
            Dropout(0.1),
            Dense(1, activation="sigmoid"),  # Salida binaria: fake o real
        ]
    )
    model.compile(
        optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


# Entrenamos el modelo
model_tf_idf = build_dense_model(input_dim=5000)
history_tf_idf = model_tf_idf.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# Evaluamos el rendimiento
y_pred_tf_idf = (model_tf_idf.predict(X_test_vec) > 0.5).astype(int)
print("\n📊 Reporte con TextVectorization (tf_idf):\n")
print(classification_report(y_test, y_pred_tf_idf, target_names=["Fake", "Real"]))

# ====================================================
# 🔹 B. MODELO CON EMBEDDINGS GloVe
# ====================================================

# Tokenizamos y convertimos texto a secuencias de enteros
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Rellenamos secuencias con ceros para que tengan la misma longitud
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen=100)
X_test_pad = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen=100)

# Cargamos embeddings GloVe (100 dimensiones)
embedding_dim = 100
embedding_index = {}
with open("Gloove/glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.strip().split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = coefs

# Creamos la matriz de embeddings con las palabras del tokenizer
word_index = tokenizer.word_index
embedding_matrix = np.zeros((5000, embedding_dim))
for word, i in word_index.items():
    if i < 5000:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Definimos el modelo con embeddings preentrenados
embedding_model = Sequential(
    [
        Embedding(
            input_dim=5000,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            trainable=False,  # No ajustamos los pesos de GloVe
        ),
        GlobalAveragePooling1D(),  # Promedia los vectores por documento
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)

embedding_model.compile(
    optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

# Entrenamos el modelo
history_embed = embedding_model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_test_pad, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# Evaluamos el modelo
y_pred_embed = (embedding_model.predict(X_test_pad) > 0.5).astype(int)
print("\nReporte con Embeddings GloVe:\n")
print(classification_report(y_test, y_pred_embed, target_names=["Fake", "Real"]))

# ====================================================
# 🔸 Visualización de resultados comparativos
# ====================================================

# Creamos un DataFrame para ambos resultados
epochs_range = range(1, 4)
results_df = pd.DataFrame(
    {
        "Época": list(epochs_range) * 2,
        "Precisión en validación": history_tf_idf.history["val_accuracy"]
        + history_embed.history["val_accuracy"],
        "Método": ["TextVectorization (TF-IDF)"] * 3 + ["GloVe Embeddings"] * 3,
    }
)

plt.figure(figsize=(10, 5))
sns.lineplot(
    data=results_df, x="Época", y="Precisión en validación", hue="Método", marker="o"
)
plt.title("Precisión de validación por técnica de vectorización")
plt.xlabel("Época de entrenamiento")
plt.ylabel("Precisión en datos de prueba")
plt.grid(True)
plt.tight_layout()
plt.show()

# 🧹 Limpieza de memoria para liberar RAM
K.clear_session()
gc.collect()
