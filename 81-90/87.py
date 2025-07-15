# ============================================
# 🧪 Ejercicio 87/200 — Visualización de atención en texto con Transformers (Keras)
# Objetivo: Comprender qué partes del texto influyen más en la decisión del modelo
# ============================================
import os
import gc
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
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
from keras import backend as K

# =============================
# ⚙️ Configuración eficiente del entorno
# =============================

num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 🧠 Configuración de lote adaptada a hardware limitado
batch_size = min(32, num_threads * 4)

# =============================
# 📥 Carga y preparación de datos
# =============================

# Lectura de los datasets
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")
df_fake["label"] = 0
df_true["label"] = 1

# Unificación y limpieza
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# División estratificada del conjunto de datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =============================
# 🧹 Preprocesamiento con TextVectorization
# =============================

# Convertimos texto en secuencias de enteros de longitud fija
vectorizer = TextVectorization(
    max_tokens=5000,
    output_sequence_length=100,
    output_mode="int",
    standardize="lower_and_strip_punctuation",
)
vectorizer.adapt(X_train)

AUTOTUNE = tf.data.AUTOTUNE

# Dataset de entrenamiento optimizado
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

# =============================
# 🧠 Definición de bloque Transformer
# =============================


def transformer_block(x, heads=2, key_dim=32, ff_dim=32, block_id=0):
    # ⚙️ Self-Attention Multi-cabeza
    attn_layer = MultiHeadAttention(
        num_heads=heads, key_dim=key_dim, name=f"mha_{block_id}"
    )
    attn_output = attn_layer(x, x)

    # 🔁 Conexión residual + normalización
    x = Add(name=f"skip_attn_{block_id}")([x, attn_output])
    x = LayerNormalization(name=f"norm_attn_{block_id}")(x)

    # 🔂 Feed-forward + conexión residual
    ff = Dense(ff_dim, activation="relu", name=f"ffn_{block_id}")(x)
    ff = Dropout(0.1)(ff)
    x = Add(name=f"skip_ffn_{block_id}")([x, ff])
    x = LayerNormalization(name=f"norm_ffn_{block_id}")(x)

    return x, attn_layer


# =============================
# 🧱 Construcción del modelo Transformer
# =============================

# Entrada: secuencias tokenizadas de 100 tokens máximo
input_layer = Input(shape=(100,), name="input_tokens")

# Embedding simple de dimensión baja (por rendimiento)
x = Embedding(input_dim=5000, output_dim=32, name="embedding")(input_layer)

# Solo 1 bloque Transformer para menor carga
x, attn_layer = transformer_block(x, block_id=0)

# Promediamos las representaciones para clasificar
x = GlobalAveragePooling1D(name="global_avg_pool")(x)
x = Dropout(0.1, name="drop_final")(x)
output = Dense(1, activation="sigmoid", name="output")(x)

model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# =============================
# 🏋️ Entrenamiento del modelo
# =============================
model.fit(train_ds, validation_data=test_ds, epochs=3, verbose=1)

# =============================
# 🧠 Interpretabilidad: extracción del bloque de atención
# =============================

# ⚠️ No se puede acceder directamente a los scores con la API funcional actual
# Requiere subclase de MultiHeadAttention para `return_attention_scores=True` (ver ejercicio 88)

# Mostramos salida del bloque de atención para un ejemplo
sample_idx = 0
for sample_batch in test_ds.take(sample_idx + 1):
    sample_input, sample_label = sample_batch
    sample_text_tensor = sample_input[:1]  # Solo una muestra
    break

# Modelo auxiliar que nos da el output del bloque de atención
attention_extractor = Model(inputs=model.input, outputs=attn_layer.output)
attn_output = attention_extractor.predict(sample_text_tensor)

print(f"\n📰 Índice: {sample_idx} | Texto vectorizado analizado")
print(f"📊 Dimensión de salida del bloque de atención: {attn_output.shape}")
print(
    "🔎 Atención visualizada indirectamente. Para scores exactos usar subclase."
)

# =============================
# ♻️ Limpieza de memoria y recursos
# =============================
del df_fake, df_true, df, X, y, X_train, X_test, y_train, y_test
gc.collect()
K.clear_session()
tf.keras.backend.clear_session()
print("✅ Recursos liberados.")
