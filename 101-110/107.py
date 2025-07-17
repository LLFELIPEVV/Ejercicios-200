# 🧠 Ejercicio 107/200 — Aplicación de LIME para explicar modelos de texto en Keras
# ========================
# 📁 1. Librerías necesarias
# ========================
import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, TextVectorization

# LIME para explicabilidad
from lime.lime_text import LimeTextExplainer

# =========================
# 🔧 2. Configuración inicial
# =========================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)

# =============================
# 📰 3. Cargar y preparar los datos
# =============================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)
df_fake["label"] = 0
df_true["label"] = 1

df = pd.concat([df_fake, df_true])[["text", "label"]]
X = df["text"].values
y = df["label"].values

# ===========================
# ✂️ 4. Separar conjuntos
# ===========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# ===================================
# 🔠 5. Vectorizar texto con Keras
# ===================================
vocab_size = 5000
max_len = 100

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(tf.convert_to_tensor(X_train))

# Convertimos a tensores
X_train_vec = vectorizer(tf.convert_to_tensor(X_train))
X_test_vec = vectorizer(tf.convert_to_tensor(X_test))

# ============================
# 🧠 6. Modelo simple
# ============================
model = Sequential(
    [
        Embedding(input_dim=vocab_size, output_dim=64),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid"),
    ]
)
model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
model.fit(
    X_train_vec, y_train, validation_data=(X_test_vec, y_test), epochs=3, batch_size=32
)

# =============================
# 🧪 7. Función para LIME
# =============================
# LIME espera recibir textos crudos (no vectores), por eso usamos vectorizer + model

class_names = ["Fake", "Real"]


def lime_predict(texts):
    # Convierte texto a tensor -> vectoriza -> predice
    vecs = vectorizer(tf.convert_to_tensor(texts))
    preds = model.predict(vecs)
    return np.concatenate([1 - preds, preds], axis=1)


# =============================
# 🔍 8. LIME en una muestra
# =============================
explainer = LimeTextExplainer(class_names=class_names)

# Escogemos una noticia al azar
idx = 5
text_to_explain = X_test[idx]
print(f"\n📰 Texto a explicar:\n{text_to_explain[:500]}...")

exp = explainer.explain_instance(text_to_explain, lime_predict, num_features=10)

# =============================
# 📊 9. Visualización con Seaborn
# =============================
weights = exp.as_list()
words, scores = zip(*weights)

# Creamos el gráfico
sns.set_palette("coolwarm")
plt.figure(figsize=(10, 5))
sns.barplot(x=scores, y=words)
plt.title("Palabras más influyentes para la predicción")
plt.xlabel("Importancia")
plt.ylabel("Palabra")
plt.grid(True)
plt.tight_layout()
plt.show()

K.clear_session()
gc.collect()
