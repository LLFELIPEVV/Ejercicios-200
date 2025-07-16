# 🧠 Ejercicio 105/200 — Guardado y exportación profesional de modelos: .h5 y .tflite con Keras

# =============================================
# 📁 1. Importar librerías necesarias
# =============================================
import os
import time
import gc
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    TextVectorization,
)

# =============================================
# 🚀 2. Configuración inicial del CPU (Ryzen 3 2200U)
# =============================================
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
tf.config.threading.set_intra_op_parallelism_threads(os.cpu_count())
tf.config.threading.set_inter_op_parallelism_threads(2)

# =============================================
# 📅 3. Cargar y procesar datos de noticias
# =============================================
df_fake = pd.read_csv("Datasets/archive/Fake.csv").dropna().sample(500, random_state=42)
df_true = pd.read_csv("Datasets/archive/True.csv").dropna().sample(500, random_state=42)
df_fake["label"] = 0  # 0 representa noticia falsa
df_true["label"] = 1  # 1 representa noticia verdadera

# Combinamos ambos conjuntos - CORREGIDO: usar lista en lugar de tupla
df = pd.concat([df_fake, df_true])[["text", "label"]]
X = df["text"].values
y = df["label"].values

# =============================================
# ⚖️ 4. Dividir en conjunto de entrenamiento y prueba
# =============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# =============================================
# 🔤 5. Vectorización de texto (convierte texto a números)
# =============================================
vocab_size = 5000  # Número máximo de palabras únicas
max_len = 100  # Longitud máxima de cada noticia

vectorizer = TextVectorization(max_tokens=vocab_size, output_sequence_length=max_len)
vectorizer.adapt(tf.convert_to_tensor(X_train))

X_train_vec = vectorizer(tf.convert_to_tensor(X_train))
X_test_vec = vectorizer(tf.convert_to_tensor(X_test))

# =============================================
# 🧶 6. Crear modelo secuencial en Keras
# =============================================
# Modelo simple con capa de embeddings y salida binaria (Fake/Real)
model = Sequential(
    [
        Embedding(
            input_dim=vocab_size, output_dim=64
        ),  # Transforma palabras en vectores densos
        GlobalAveragePooling1D(),  # Promedia los vectores
        Dense(1, activation="sigmoid"),  # Capa de salida binaria
    ]
)

model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])

# Entrenamos el modelo por 3 épocas
history = model.fit(
    X_train_vec, y_train, validation_data=(X_test_vec, y_test), epochs=3, batch_size=32
)

# =============================================
# 💾 7. Guardar el modelo completo como archivo .h5
# =============================================
model.save("modelo_fake_news.h5")
print("\n✅ Modelo guardado como .h5")

# =============================================
# 🔄 8. Convertir modelo a formato optimizado .tflite
# =============================================
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("modelo_fake_news.tflite", "wb") as f:
    f.write(tflite_model)
print("\n✅ Modelo exportado como .tflite")

# =============================================
# ⏱️ 9. Medir tiempo de inferencia con ambos formatos
# =============================================
print("\n⏱️ Inference time (.h5):")
start = time.time()
_ = model.predict(X_test_vec)
print("%.4f segundos" % (time.time() - start))

# Para usar el modelo .tflite correctamente necesitamos el intérprete
print("\n⏱️ Inference time (.tflite):")
interpreter = tf.lite.Interpreter(model_path="modelo_fake_news.tflite")
interpreter.allocate_tensors()

# Obtener información de entrada y salida
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

start = time.time()
# Procesar cada muestra individualmente para TFLite
for i in range(len(X_test_vec)):
    interpreter.set_tensor(input_details[0]["index"], X_test_vec[i : i + 1])
    interpreter.invoke()
    _ = interpreter.get_tensor(output_details[0]["index"])
print("%.4f segundos" % (time.time() - start))

# =============================================
# 📊 10. Visualizar curva de precisión por época
# =============================================
sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 4))
sns.lineplot(
    x=range(1, 4),
    y=history.history["val_accuracy"],
    marker="o",
    label="Precisión Validación",
)
sns.lineplot(
    x=range(1, 4),
    y=history.history["accuracy"],
    marker="s",
    label="Precisión Entrenamiento",
)
plt.title("Evolución de precisión por época")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# ♻️ 11. Liberar memoria del modelo
# =============================================
K.clear_session()
gc.collect()
