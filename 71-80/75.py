# ✅ Ejercicio 75/200 — Clasificación de Fake News con Self-Attention (Transformer básico)
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input,
    Embedding,
    Dropout,
    Dense,
    TextVectorization,
    MultiHeadAttention,
    Add,
    LayerNormalization,
    GlobalAveragePooling1D,
)

# ---------------------------------------------
# ⚙️ Configuración dinámica de hardware (CPU)
# ---------------------------------------------
num_threads = os.cpu_count()

os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

if num_threads <= 4:
    batch_size = 32
elif num_threads <= 8:
    batch_size = 64
else:
    batch_size = 128

# ---------------------------------------------
# 📥 Carga de datos
# ---------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ---------------------------------------------
# 🔠 Vectorización
# ---------------------------------------------
vectorizer = TextVectorization(
    max_tokens=5000,  # Vocabulario más pequeño
    output_sequence_length=100,  # Recorte de longitud para más rapidez
    output_mode="int",
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(512)
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


# ---------------------------------------------
# 🔁 Transformer encoder simplificado
# ---------------------------------------------
def fast_transformer_block(x, num_heads=2, key_dim=16, ff_dim=32, dropout_rate=0.1):
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn])
    x = LayerNormalization()(x)

    ff = Dense(ff_dim, activation="relu")(x)
    ff = Dropout(dropout_rate)(ff)
    x = Add()([x, ff])
    x = LayerNormalization()(x)

    return x


# ---------------------------------------------
# 🏗️ Modelo
# ---------------------------------------------
input_layer = Input(shape=(100,))
x = Embedding(input_dim=5000, output_dim=32)(input_layer)
x = fast_transformer_block(x)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.1)(x)
output_layer = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# ---------------------------------------------
# 🏋️ Entrenamiento (rápido)
# ---------------------------------------------
model.fit(train_ds, validation_data=test_ds, epochs=3, verbose=1)

# ---------------------------------------------
# 📊 Evaluación
# ---------------------------------------------
y_pred = model.predict(test_ds).flatten()
y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación:")
print(classification_report(y_true, y_pred_labels, zero_division=0))
