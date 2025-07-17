# 🧠 Ejercicio 112/200 – Preparación de datos eficiente con tf.data.Dataset para detección de noticias falsas
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import (
    TextVectorization,
    Embedding,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
)

# Simulamos un pequeño conjunto de noticias reales y falsas
data = {
    "text": [
        "Aliens replaced the president",
        "NASA finds water on Mars",
        "Vaccines are harmful, say sources",
        "Election process was fair",
        "5G spreads viruses",
        "Economy shows signs of recovery",
    ],
    "label": [1, 0, 1, 0, 1, 0],  # 1: Fake, 0: Real
}

df = pd.DataFrame(data)

# Dividimos los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    df["text"].values,
    np.array(df["label"]).astype("float32"),
    test_size=0.3,
    random_state=42,
)

# Paso 1: Preprocesamiento con TextVectorization
max_tokens = 1000
sequence_length = 30

vectorizer = TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation",
)

# Adaptamos el vectorizador al vocabulario del conjunto de entrenamiento
vectorizer.adapt(X_train)

# Paso 2: Crear datasets con tf.data.Dataset
# Convertimos los arrays en objetos Dataset
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

# Paso 3: Aplicamos operaciones de transformación al flujo
AUTOTUNE = tf.data.AUTOTUNE


# Esta función transforma el texto en tokens enteros
def vectorize_text(text, label):
    return vectorizer(text), label


# Aplicamos transformaciones al dataset
train_ds = (
    train_ds.map(vectorize_text, num_parallel_calls=AUTOTUNE)  # Aplica la tokenización
    .cache()  # Guarda en RAM para reutilizar
    .shuffle(buffer_size=16)  # Mezcla el orden (para entrenamiento)
    .batch(2)  # Agrupa en lotes de 2
    .prefetch(buffer_size=AUTOTUNE)  # Precarga el siguiente batch
)

test_ds = test_ds.map(vectorize_text).batch(2).cache().prefetch(AUTOTUNE)

# Paso 4: Crear un modelo simple
model = Sequential(
    [
        Embedding(input_dim=max_tokens, output_dim=16),
        GlobalAveragePooling1D(),
        Dense(8, activation="relu"),
        Dropout(0.4),
        Dense(1, activation="sigmoid"),
    ]
)

# Compilamos el modelo
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callbacks para evitar sobreentrenamiento y reducir learning rate automáticamente
early_stop = EarlyStopping(patience=3, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=2)

# Entrenamos el modelo
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=15,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# Convertimos resultados de entrenamiento a DataFrame
hist_df = pd.DataFrame(history.history)

# Estilo visual
sns.set(style="whitegrid")

# Graficar precisión
plt.figure(figsize=(10, 5))
sns.lineplot(data=hist_df["accuracy"], label="Entrenamiento", linewidth=2.5)
sns.lineplot(data=hist_df["val_accuracy"], label="Validación", linewidth=2.5)
plt.title("Precisión del modelo en cada época", fontsize=14)
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.show()

# Graficar pérdida
plt.figure(figsize=(10, 5))
sns.lineplot(data=hist_df["loss"], label="Entrenamiento", linewidth=2.5)
sns.lineplot(data=hist_df["val_loss"], label="Validación", linewidth=2.5)
plt.title("Pérdida del modelo en cada época", fontsize=14)
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.show()
