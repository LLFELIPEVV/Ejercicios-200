# 🧠 Ejercicio 100/200 — Comparación visual y explicativa: LSTM vs Atención (mini Transformer)

# ========== Importación de librerías ==========
import os
import gc
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from keras.models import Model
from keras.layers import (
    Input,
    Embedding,
    LSTM,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Add,
    TextVectorization,
)
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ========== Configuración del entorno ==========
# Limitamos los hilos del procesador para mejorar el rendimiento en un CPU modesto como Ryzen 3 2200U.
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)
batch_size = 32  # Tamaño de lote pequeño para evitar saturación del CPU

# ========== Carga y limpieza de datos ==========
# Cargamos noticias verdaderas y falsas. Luego les asignamos etiquetas (1 = real, 0 = fake).
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna()
df_true = pd.read_csv("Datasets/archive/True.csv").dropna()
df_fake["label"] = 0
df_true["label"] = 1

# Combinamos ambas fuentes y seleccionamos solo 1000 muestras para acelerar el proceso
df = pd.concat([df_fake, df_true])[["text", "label"]].sample(1000, random_state=42)
X, y = df["text"].values, df["label"].values

# Dividimos los datos en 80% entrenamiento y 20% prueba, manteniendo proporción de clases
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ========== Vectorización del texto ==========
# Convertimos el texto en secuencias de números. Cada palabra tendrá un número asociado.
vocab_size = 5000  # Tamaño máximo del vocabulario
max_len = 100  # Longitud máxima de cada texto (relleno si es más corto)

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(
    tf.convert_to_tensor(X_train)
)  # Aprende el vocabulario de entrenamiento

# Aplicamos el vectorizador a los textos
X_train_vec = vectorizer(tf.convert_to_tensor(X_train))
X_test_vec = vectorizer(tf.convert_to_tensor(X_test))

# ========== Modelo LSTM (Long Short-Term Memory) ==========
# Este modelo recorre el texto secuencialmente, recordando el contexto anterior
input_lstm = Input(shape=(max_len,))
embed = Embedding(input_dim=vocab_size, output_dim=64)(
    input_lstm
)  # Convierte palabras a vectores de 64 dimensiones
lstm = LSTM(64)(embed)  # Capa LSTM que "recuerda" información pasada
drop = Dropout(0.2)(lstm)  # Previene sobreajuste desactivando nodos aleatorios
out = Dense(1, activation="sigmoid")(drop)  # Salida: probabilidad de que sea real

model_lstm = Model(input_lstm, out)
model_lstm.compile(
    optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

# Entrenamiento del modelo LSTM
history_lstm = model_lstm.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# Evaluamos el modelo con los datos de prueba
y_pred_lstm = (model_lstm.predict(X_test_vec) > 0.5).astype(int)
print("\nLSTM Reporte:\n")
print(classification_report(y_test, y_pred_lstm, target_names=["Fake", "Real"]))

# ========== Modelo con Atención (transformer mini) ==========
# Este modelo decide qué partes del texto son más importantes usando un mecanismo de atención
input_attn = Input(shape=(max_len,))
embed = Embedding(input_dim=vocab_size, output_dim=64)(input_attn)  # Mismo embedding
attn = MultiHeadAttention(num_heads=2, key_dim=64)(
    embed, embed
)  # La atención compara cada palabra con todas las demás
add = Add()([embed, attn])  # Sumamos entrada original + atención (residual connection)
norm = LayerNormalization()(add)  # Normalizamos para estabilidad numérica
gap = GlobalAveragePooling1D()(
    norm
)  # Promediamos la información a lo largo de toda la secuencia
drop = Dropout(0.2)(gap)
out = Dense(1, activation="sigmoid")(drop)

model_attn = Model(input_attn, out)
model_attn.compile(
    optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

# Entrenamiento del modelo con atención
history_attn = model_attn.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# Evaluación del modelo de atención
y_pred_attn = (model_attn.predict(X_test_vec) > 0.5).astype(int)
print("\nAtención Reporte:\n")
print(classification_report(y_test, y_pred_attn, target_names=["Fake", "Real"]))

# ========== Visualización de resultados ==========
plt.figure(figsize=(10, 5))
sns.set_style("whitegrid")
sns.lineplot(
    x=range(1, 4),
    y=history_lstm.history["val_accuracy"],
    label="LSTM",
    marker="o",
    linewidth=2,
)
sns.lineplot(
    x=range(1, 4),
    y=history_attn.history["val_accuracy"],
    label="Atención",
    marker="s",
    linewidth=2,
)
plt.title("Precisión de validación: LSTM vs Atención", fontsize=14)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Precisión", fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# ========== Limpieza de memoria ==========
# Libera recursos del modelo para evitar problemas si se entrena otro luego
K.clear_session()
gc.collect()
