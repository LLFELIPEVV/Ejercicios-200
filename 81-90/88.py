# 🧪 Ejercicio 88/200 — Visualización de pesos en capas densas: Interpretabilidad básica en redes neuronales densas con Keras
# ---------------------------------------------------
# 🧠 Interpretabilidad de capas densas en texto con TF-IDF + Keras
# ---------------------------------------------------
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# ---------------------------------------------------
# ⚙️ Configuración del entorno y rendimiento del CPU
# ---------------------------------------------------
num_threads = os.cpu_count()
os.environ["OMP_NUM_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTRAOP_THREADS"] = str(num_threads)
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(2)

# 🔧 Configura el tamaño de lote según la capacidad de tu equipo
batch_size = 32 if num_threads <= 4 else 64 if num_threads <= 8 else 128

# ---------------------------------------------------
# 📥 Carga y preparación de datos (Fake vs Real news)
# ---------------------------------------------------
df_fake = pd.read_csv("Datasets/archive/Fake.csv")
df_true = pd.read_csv("Datasets/archive/True.csv")

# Etiquetado binario
df_fake["label"] = 0
df_true["label"] = 1

# Unificación y limpieza básica
df = pd.concat([df_fake, df_true], ignore_index=True)[["text", "label"]].dropna()
X, y = df["text"].values, df["label"].values

# División estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------
# 🔤 Vectorización TF-IDF
# ---------------------------------------------------
vectorizer = TfidfVectorizer(
    max_features=5000,  # Reducido por eficiencia
    stop_words="english",
)
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# ---------------------------------------------------
# 🧱 Definición del modelo Keras (Denso simple)
# ---------------------------------------------------
model = Sequential(
    [
        Input(shape=(5000,)),  # Vector TF-IDF como entrada
        Dense(32, activation="relu"),  # Capa oculta con 32 unidades
        Dropout(0.1),  # Prevención de sobreajuste
        Dense(1, activation="sigmoid"),  # Capa de salida binaria
    ]
)

model.compile(
    optimizer=Adam(1e-3),  # Optimizador eficiente
    loss="binary_crossentropy",  # Tarea de clasificación binaria
    metrics=["accuracy"],
)

# ---------------------------------------------------
# 🏋️ Entrenamiento del modelo
# ---------------------------------------------------
model.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=3,
    batch_size=batch_size,
    verbose=1,
)

# ---------------------------------------------------
# 📈 Evaluación del rendimiento del modelo
# ---------------------------------------------------
y_pred = (model.predict(X_test_vec) > 0.5).astype(int).flatten()

print("\n📊 Reporte de clasificación:\n")
print(
    classification_report(
        y_test, y_pred, target_names=["Fake", "Real"], zero_division=0
    )
)

# ---------------------------------------------------
# 🔍 Interpretabilidad: Análisis de pesos por palabra
# ---------------------------------------------------

# Vocabulario procesado por el vectorizador
feature_names = vectorizer.get_feature_names_out()

# Extracción de los pesos de la primera capa densa (matriz [tokens, neuronas])
weights = model.layers[0].get_weights()[0]

# Calculamos el peso promedio por palabra (media entre neuronas)
mean_weights = weights.mean(axis=1)

# 🟢 Palabras más fuertemente asociadas a la clase "Real"
top_positive_indices = np.argsort(mean_weights)[-15:]

# 🔴 Palabras más asociadas a la clase "Fake"
top_negative_indices = np.argsort(mean_weights)[:15]

# ---------------------------------------------------
# 📊 Visualización de características más influyentes
# ---------------------------------------------------
plt.figure(figsize=(12, 6))
plt.barh(
    [feature_names[i] for i in top_positive_indices],
    mean_weights[top_positive_indices],
    color="green",
)
plt.title("Palabras más asociadas a noticias reales")
plt.xlabel("Peso promedio (importancia)")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
plt.barh(
    [feature_names[i] for i in top_negative_indices],
    mean_weights[top_negative_indices],
    color="red",
)
plt.title("Palabras más asociadas a noticias falsas")
plt.xlabel("Peso promedio (importancia)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------------
# ♻️ Limpieza de memoria y recursos
# ---------------------------------------------------
del X_train_vec, X_test_vec, weights, vectorizer, feature_names
gc.collect()
K.clear_session()
