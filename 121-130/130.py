# ✅ Ejercicio 130/200: Visualización y Exportación Optimizada de un Modelo de Fake News
# ---------------------------------------------------------
# 🔎 Importación de librerías necesarias para el proyecto
# ---------------------------------------------------------
import tensorflow as tf  # Librería para construir modelos de Deep Learning
import re  # Librería para usar expresiones regulares y limpiar texto
import pandas as pd  # Para manipular datos en tablas
import seaborn as sns  # Para crear gráficos estadísticos
import matplotlib.pyplot as plt  # Para mostrar los gráficos

from keras.models import Sequential  # Tipo de modelo lineal secuencial
from keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
)  # Callbacks útiles para optimizar el entrenamiento
from keras.layers import (
    TextVectorization,
    Dense,
    GlobalAveragePooling1D,
    Embedding,
)  # Capas del modelo

# ---------------------------------------------------------
# 📊 1. Dataset de ejemplo con dos clases (Fake = 1, Real = 0)
# ---------------------------------------------------------
data = {
    "text": [
        "Breaking news: aliens found on Mars!",
        "Government releases economic growth data",
        "Shocking! Cure for cancer found in herbs!",
        "Central bank maintains interest rate",
        "Elon Musk buys the Moon?",
        "Stock market shows steady recovery",
    ],
    "label": [1, 0, 1, 0, 1, 0],  # Etiquetas: 1 = Fake News, 0 = Noticias reales
}

df = pd.DataFrame(data)  # Creamos una tabla con pandas

# ---------------------------------------------------------
# 📊 2. Visualización de distribución de clases
# ---------------------------------------------------------
plt.figure(figsize=(6, 4))  # Tamaño del gráfico
sns.countplot(x="label", data=df, palette="coolwarm")  # Conteo por clase
plt.title("Distribución de Clases: Fake News (1) vs Reales (0)")
plt.xlabel("Clase")
plt.ylabel("Cantidad de ejemplos")
plt.xticks([0, 1], ["Real (0)", "Fake (1)"])
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 🧹 3. Limpieza de texto con expresiones regulares
# ---------------------------------------------------------
def clean_text(text):
    """
    Esta función transforma el texto:
    - A minúsculas
    - Elimina URLs, signos de puntuación, caracteres especiales
    - Normaliza espacios en blanco
    """
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r"https?\S+", "", text)  # Eliminar URLs
    text = re.sub(r"[^a-z\s]", "", text)  # Eliminar todo excepto letras y espacios
    text = re.sub(r"\s+", " ", text).strip()  # Espacios extra
    return text


# Aplicamos la función a cada fila de texto
df["clean"] = df["text"].apply(clean_text)

# ---------------------------------------------------------
# ✂️ 4. Tokenización avanzada con TextVectorization de Keras
# ---------------------------------------------------------
max_tokens = 1000  # Número máximo de palabras únicas (vocabulario)
seq_len = 20  # Longitud fija de cada texto (en palabras)

# Vectorizador personalizado
vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_sequence_length=seq_len,
    standardize=None,  # Ya limpiamos manualmente
)

vectorizer.adapt(df["clean"])  # Aprende el vocabulario

# ---------------------------------------------------------
# 🧠 5. Vectorizar texto y convertir etiquetas a tensores
# ---------------------------------------------------------
X = vectorizer(df["clean"])  # Convierte texto a secuencias de números
y = tf.convert_to_tensor(df["label"])  # Convierte etiquetas a tensor

# ---------------------------------------------------------
# 🔀 6. División en entrenamiento (4 ejemplos) y validación (2 ejemplos)
# ---------------------------------------------------------
X_train, X_val = X[:4], X[4:]
y_train, y_val = y[:4], y[4:]

# ---------------------------------------------------------
# 🏗️ 7. Creación de un modelo ligero secuencial
# ---------------------------------------------------------
model = Sequential(
    [
        Embedding(
            input_dim=max_tokens, output_dim=8
        ),  # Capa de embedding (palabras → vectores)
        GlobalAveragePooling1D(),  # Saca un promedio de todos los vectores del texto
        Dense(1, activation="sigmoid"),  # Capa final para clasificar entre 0 y 1
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ---------------------------------------------------------
# 🛑 8. Callbacks: para entrenamiento más eficiente
# ---------------------------------------------------------
early_stop = EarlyStopping(
    patience=3, restore_best_weights=True
)  # Detiene si no mejora
reduce_lr = ReduceLROnPlateau(patience=2)  # Reduce tasa de aprendizaje si no mejora

# ---------------------------------------------------------
# 🏋️ 9. Entrenamiento del modelo
# ---------------------------------------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ---------------------------------------------------------
# 🧊 10. Congelar el modelo (no más entrenamiento)
# ---------------------------------------------------------
for layer in model.layers:
    layer.trainable = False  # Todas las capas congeladas

print(
    "✅ Todas las capas están congeladas:", all([not l.trainable for l in model.layers])
)

# ---------------------------------------------------------
# 💾 11. Guardar el modelo en formato .h5 (Keras)
# ---------------------------------------------------------
model.save("modelo_fake_news.h5")

# ---------------------------------------------------------
# 💡 12. Conversión a .tflite para producción en dispositivos CPU
# ---------------------------------------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Compresión ligera
tflite_model = converter.convert()

with open("modelo_fake_news.tflite", "wb") as f:
    f.write(tflite_model)

print("📦 Modelo exportado correctamente en formatos .h5 y .tflite")


# ---------------------------------------------------------
# 📈 Gráfica de entrenamiento y validación
# ---------------------------------------------------------
plt.figure(figsize=(10, 4))

# Precisión
plt.subplot(1, 2, 1)
sns.lineplot(data=history.history["accuracy"], label="Entrenamiento")
sns.lineplot(data=history.history["val_accuracy"], label="Validación")
plt.title("Precisión")
plt.xlabel("Épocas")
plt.ylabel("Precisión")
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
sns.lineplot(data=history.history["loss"], label="Entrenamiento")
sns.lineplot(data=history.history["val_loss"], label="Validación")
plt.title("Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()

plt.tight_layout()
plt.show()
