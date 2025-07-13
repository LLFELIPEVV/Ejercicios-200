# ✅ Ejercicio 80/200 — Fine-Tuning de Word Embeddings Preentrenados (GloVe) para Fake News
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import (
    Input,
    Embedding,
    TextVectorization,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
)

# --------------------------------------------------
# ⚙️ Configuración de rendimiento multihilo
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# --------------------------------------------------
# 📥 Carga de datos y etiquetado
# --------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

# Unimos y seleccionamos las columnas necesarias
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# División de datos con estratificación (misma proporción de clases en train/test)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

# --------------------------------------------------
# 🧹 Preprocesamiento de texto con TextVectorization
# --------------------------------------------------
max_vocab = 5000  # Tamaño del vocabulario
seq_len = 100  # Longitud fija de secuencia

vectorizer = TextVectorization(
    max_tokens=max_vocab,
    output_sequence_length=seq_len,
    output_mode="int",
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)

# Creamos diccionario {palabra: índice}
vocab = vectorizer.get_vocabulary()
word_index = {word: idx for idx, word in enumerate(vocab)}

# --------------------------------------------------
# 💾 Carga de embeddings GloVe
# --------------------------------------------------
embedding_dim = 100
embedding_index = {}

# Leer archivo GloVe línea por línea
with open("Gloove/glove.6B.100d.txt", encoding="utf-8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        embedding_index[word] = vector

# Construcción de la matriz de embeddings (vocabulario del modelo)
embedding_matrix = np.zeros((max_vocab, embedding_dim))
for word, idx in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[idx] = embedding_vector
    # Si no se encuentra la palabra, se deja el vector como ceros

# --------------------------------------------------
# 🧠 Modelo secuencial con Embedding GloVe ajustable
# --------------------------------------------------
model = Sequential(name="GloVe_FineTuned_LSTM")
model.add(Input(shape=(seq_len,), name="input_tokens"))
model.add(
    Embedding(
        input_dim=max_vocab,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=True,  # ✅ Permite fine-tuning
        name="embedding_glove_finetune",
    )
)
model.add(Bidirectional(LSTM(32), name="bilstm"))
model.add(Dropout(0.1, name="dropout"))
model.add(Dense(1, activation="sigmoid", name="output"))

model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# --------------------------------------------------
# 🔁 Preparación del dataset con tf.data
# --------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(buffer_size=512)
    .batch(batch_size)
    .map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# --------------------------------------------------
# 🏋️ Entrenamiento
# --------------------------------------------------
model.fit(train_ds, validation_data=test_ds, epochs=3)

# --------------------------------------------------
# 📊 Evaluación
# --------------------------------------------------
y_pred = model.predict(test_ds).flatten()
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación con embeddings GloVe fine-tuned:")
print(classification_report(y_true, y_pred_labels, zero_division=0))
