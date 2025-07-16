# 🧠 Ejercicio 103/200 — Regularización con L2 (Keras): Controlando el sobreajuste en redes densas para clasificación de noticias
import os
import gc
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input

# ========================
# 🔧 Configuración del CPU
# ========================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)
batch_size = 32

# ======================
# 📥 Cargamos los datos
# ======================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)
df_fake["label"] = 0
df_true["label"] = 1

# Combinamos noticias falsas y reales
df = pd.concat([df_fake, df_true], ignore_index=True)
X = df["text"].values
y = df["label"].values

# ============================
# ✂️ Separar en train/test
# ============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# =======================================
# 🔠 Transformamos texto con TF-IDF
# =======================================
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train).toarray()
X_test_vec = vectorizer.transform(X_test).toarray()

# =============================
# 🧱 Modelo SIN regularización
# =============================
model_simple = Sequential(
    [
        Input(shape=(5000,)),
        Dense(32, activation="relu"),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model_simple.compile(
    optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"]
)

history_simple = model_simple.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=5,
    batch_size=batch_size,
    verbose=1,
)

# =============================
# 🧱 Modelo CON L2 regularization
# =============================
model_l2 = Sequential(
    [
        Input(shape=(5000,)),
        Dense(32, activation="relu", kernel_regularizer=l2(0.01)),
        Dropout(0.1),
        Dense(1, activation="sigmoid"),
    ]
)
model_l2.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

history_l2 = model_l2.fit(
    X_train_vec,
    y_train,
    validation_data=(X_test_vec, y_test),
    epochs=5,
    batch_size=batch_size,
    verbose=1,
)

# ================================
# 📊 Comparación visual con seaborn
# ================================
plt.figure(figsize=(12, 6))

sns.lineplot(
    x=range(1, 6), y=history_simple.history["val_accuracy"], label="Sin L2", marker="o"
)
sns.lineplot(
    x=range(1, 6), y=history_l2.history["val_accuracy"], label="Con L2", marker="s"
)
plt.title("Precisión de validación con y sin regularización L2")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ========================
# 📋 Reportes finales
# ========================
y_pred_simple = (model_simple.predict(X_test_vec) > 0.5).astype(int)
y_pred_l2 = (model_l2.predict(X_test_vec) > 0.5).astype(int)

print("\n🧾 Reporte sin regularización:\n")
print(classification_report(y_test, y_pred_simple, target_names=["Fake", "Real"]))

print("\n🧾 Reporte con L2 regularization:\n")
print(classification_report(y_test, y_pred_l2, target_names=["Fake", "Real"]))

# ========================
# 🧹 Limpieza de memoria
# ========================
K.clear_session()
gc.collect()
