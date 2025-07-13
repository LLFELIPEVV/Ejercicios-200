# ✅ Ejercicio 77/200 — Transformer Profundo para Fake News (Stacked Self-Attention)
import os
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Model
from keras.layers import (
    Input,
    TextVectorization,
    Embedding,
    LayerNormalization,
    MultiHeadAttention,
    Dense,
    Dropout,
    Add,
    GlobalAveragePooling1D,
)
from keras.optimizers import Adam

# --------------------------------------------------
# ⚙️ Configuración dinámica para uso eficiente del CPU
# --------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# Tamaño de lote adaptado al hardware
batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# --------------------------------------------------
# 📥 Carga y preparación de datos
# --------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --------------------------------------------------
# 🧹 Preprocesamiento: Tokenización con vectorizador
# --------------------------------------------------
vectorizer = TextVectorization(
    max_tokens=5000,  # Vocabulario limitado
    output_sequence_length=100,  # Longitud fija de entrada
    output_mode="int",
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)

AUTOTUNE = tf.data.AUTOTUNE

# Dataset de entrenamiento con pipeline optimizado
train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .shuffle(512)
    .batch(batch_size)
    .map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)

# Dataset de prueba
test_ds = (
    tf.data.Dataset.from_tensor_slices((X_test, y_test))
    .batch(batch_size)
    .map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
    .prefetch(AUTOTUNE)
)


# --------------------------------------------------
# 🧠 Definición de un bloque Transformer básico
# --------------------------------------------------
def transformer_block(
    x, num_heads=2, key_dim=32, ff_dim=32, dropout_rate=0.1, block_id=0
):
    attn_output = MultiHeadAttention(
        num_heads=num_heads, key_dim=key_dim, name=f"mha_{block_id}"
    )(x, x)
    x = Add(name=f"skip_connection_attention_{block_id}")([x, attn_output])
    x = LayerNormalization(name=f"norm_attention_{block_id}")(x)

    ff = Dense(ff_dim, activation="relu", name=f"ffn_{block_id}")(x)
    ff = Dropout(dropout_rate, name=f"drop_ffn_{block_id}")(ff)
    x = Add(name=f"skip_connection_ffn_{block_id}")([x, ff])
    x = LayerNormalization(name=f"norm_ffn_{block_id}")(x)
    return x


# --------------------------------------------------
# 🧱 Construcción del modelo completo
# --------------------------------------------------
input_layer = Input(shape=(100,), name="input_tokens")
x = Embedding(input_dim=5000, output_dim=32, name="embedding")(input_layer)

# 🔁 Apilamos 3 bloques Transformer
for i in range(3):
    x = transformer_block(
        x,
        num_heads=2,
        key_dim=32,
        ff_dim=32,
        dropout_rate=0.1,
        block_id=i,  # Nuevo argumento para nombres únicos
    )

x = GlobalAveragePooling1D(name="global_avg_pool")(
    x
)  # Resumen vectorial de la secuencia
x = Dropout(0.1, name="drop_final")(x)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

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

print("\n📈 Reporte de clasificación:")
print(classification_report(y_true, y_pred_labels, zero_division=0))
