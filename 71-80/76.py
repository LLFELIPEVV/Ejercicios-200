# ✅ Ejercicio 76/200 — Implementar Positional Encoding manualmente en un modelo Transformer para clasificación de Fake News
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import (
    Input,
    TextVectorization,
    Layer,
    MultiHeadAttention,
    Add,
    LayerNormalization,
    Dense,
    Dropout,
    Embedding,
    GlobalAveragePooling1D,
)

# --------------------------------------------------
# ⚙️ Configuración dinámica de uso de hilos para CPU
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# -----------------------------
# 📥 Carga y preprocesamiento de datos
# -----------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------
# 🧹 Vectorización
# -----------------------------
vectorizer = TextVectorization(
    max_tokens=5000,
    output_sequence_length=100,
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


# -----------------------------
# 📐 Positional Encoding Layer personalizada
# -----------------------------
class PositionalEncoding(Layer):
    """Agrega codificación de posición sinusoidal al embedding."""

    def __init__(self, sequence_len, d_model):
        super().__init__()
        self.pos_encoding = self._generate_positional_encoding(sequence_len, d_model)

    def _generate_positional_encoding(self, seq_len, d_model):
        pos = np.arange(seq_len)[:, np.newaxis]
        i = np.arange(d_model)[np.newaxis, :]
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        angle_rads = pos * angle_rates

        # Aplicar seno a índices pares y coseno a impares
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]


# -----------------------------
# 🧠 Bloque Transformer simple
# -----------------------------
def transformer_block(x, num_heads, key_dim):
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    x = Add()([x, attn_output])
    x = LayerNormalization()(x)

    ff = Dense(32, activation="relu")(x)
    ff = Dropout(0.1)(ff)
    x = Add()([x, ff])
    return LayerNormalization()(x)


# -----------------------------
# 🔨 Construcción del modelo
# -----------------------------
input_layer = Input(shape=(100,))
embedding_dim = 32  # 💡 Alineado con el PositionalEncoding

x = Embedding(input_dim=5000, output_dim=embedding_dim)(input_layer)
x = PositionalEncoding(sequence_len=100, d_model=embedding_dim)(x)
x = transformer_block(x, num_heads=2, key_dim=16)
x = GlobalAveragePooling1D()(x)
x = Dropout(0.3)(x)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# -----------------------------
# 🏋️ Entrenamiento
# -----------------------------
model.fit(train_ds, validation_data=test_ds, epochs=3)

# -----------------------------
# 📊 Evaluación
# -----------------------------
y_pred = model.predict(test_ds).flatten()
y_true = np.concatenate([y for _, y in test_ds], axis=0)
y_pred_labels = (y_pred > 0.5).astype(int)

print("\n📈 Reporte de clasificación:")
print(classification_report(y_true, y_pred_labels, zero_division=0))
