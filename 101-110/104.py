# 🧠 Ejercicio 104/200 — Preparación eficiente con tf.data.Dataset y optimización ligera de inferencia
import os
import gc
import time
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
)

# ======================
# 🧠 Configuración básica
# ======================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)
batch_size = 32
max_len = 100
vocab_size = 5000

# =================================
# 📥 Carga y preparación de dataset
# =================================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)
df_fake["label"] = 0
df_true["label"] = 1
df = pd.concat([df_fake, df_true])[["text", "label"]].sample(1000, random_state=42)

# ⚠️ Filtramos textos vacíos para evitar errores con tf.data
df = df[df["text"].str.strip().astype(bool)]

X = df["text"].values
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# 🔠 Vectorizador de texto
# =========================
vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(tf.convert_to_tensor(X_train))

# =======================================
# 🔁 Dataset con listas (forma clásica)
# =======================================
X_train_vec = vectorizer(tf.convert_to_tensor(X_train))
X_test_vec = vectorizer(tf.convert_to_tensor(X_test))


# =======================================
# 🔁 Dataset optimizado con tf.data.Dataset
# =======================================
def preprocess(text, label):
    return vectorizer(text), label


train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(preprocess).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# ============================
# 🧱 Modelo simple y eficiente
# ============================
model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=64),
        GlobalAveragePooling1D(),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

# ======================
# 🧪 Entrenamiento
# ======================
history = model.fit(train_ds, validation_data=test_ds, epochs=3, verbose=1)

# ======================
# ⏱️ Medición de inferencia
# ======================
print("\n⏱️ Inference time with .predict (Lista):")
start = time.time()
_ = model.predict(X_test_vec, batch_size=batch_size)
print("Tiempo con lista:", round(time.time() - start, 2), "s")

print("\n⏱️ Inference time with tf.data.Dataset:")
start_time = time.time()
_ = model.predict(test_ds)
end_time = time.time()
print(f"⏱️ Inference time (.predict con tensor): {end_time - start_time:.4f} segundos")

# ======================
# 📊 Reporte de métricas
# ======================
y_pred = (model.predict(test_ds) > 0.5).astype(int)
print("\n📈 Reporte final con tf.data.Dataset:\n")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# ======================
# 📈 Visualización de entrenamiento
# ======================
plt.figure(figsize=(10, 5))
sns.lineplot(data=history.history["accuracy"], label="Train", marker="o")
sns.lineplot(data=history.history["val_accuracy"], label="Validation", marker="s")
plt.title("Precisión del modelo usando tf.data.Dataset")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()

# 🧹 Limpieza
K.clear_session()
gc.collect()
