# 🧪 Ejercicio 113/200 — Optimización del entrenamiento con EarlyStopping y ReduceLROnPlateau en un modelo liviano de texto

# 📚 Importamos las bibliotecas necesarias
import re  # para limpiar el texto usando expresiones regulares
import pandas as pd  # para manejar datos en forma de tabla
import seaborn as sns  # para graficar resultados de entrenamiento
import matplotlib.pyplot as plt  # para mostrar los gráficos

import tensorflow as tf  # framework principal de Deep Learning
from sklearn.model_selection import (
    train_test_split,
)  # para dividir el dataset en entrenamiento y prueba
from keras.models import Sequential  # para crear el modelo capa por capa
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)  # callbacks de optimización
from keras.layers import (
    TextVectorization,
    Dense,
    Embedding,
    GlobalAveragePooling1D,
)  # capas del modelo

# ✅ 1. Creamos un conjunto de datos simulado de noticias
data = {
    "text": [
        "Breaking news: President caught lying!",
        "NASA discovers water on Mars!",
        "You won't believe what this dog did...",
        "Click here to win $1000!",
        "Scientists prove Earth is round.",
        "Miracle cure for baldness found!",
        "Government hides the truth again!",
        "Facebook down for 3 hours.",
    ],
    "label": [1, 0, 1, 1, 0, 1, 1, 0],  # 1 = noticia falsa (fake), 0 = noticia real
}

df = pd.DataFrame(data)


# ✅ 2. Limpiamos el texto (eliminamos signos, URLs, y pasamos a minúsculas)
def clean_text(text):
    text = text.lower()  # convertimos a minúsculas
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # eliminamos URLs
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # eliminamos signos de puntuación
    return text.strip()


df["text"] = df["text"].apply(clean_text)

# ✅ 3. Dividimos el dataset en entrenamiento (75%) y prueba (25%)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42
)

# ✅ 4. Convertimos el texto en números (tokenización) con TextVectorization
vectorizer = TextVectorization(
    max_tokens=1000,  # máximo número de palabras únicas
    output_mode="int",  # salida como secuencia de enteros
    output_sequence_length=20,  # longitud fija para todas las secuencias
)
vectorizer.adapt(X_train)  # el vectorizador aprende del texto de entrenamiento

# ✅ 5. Creamos datasets eficientes con tf.data
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))


# Función que aplica la vectorización a cada texto
def vectorize_text(text, label):
    return vectorizer(text), label


# Aplicamos la función al dataset y optimizamos su carga
train_ds = train_ds.map(vectorize_text).batch(2).cache().prefetch(1)
test_ds = test_ds.map(vectorize_text).batch(2).cache().prefetch(1)

# ✅ 6. Definimos un modelo de red neuronal pequeño y eficiente
model = Sequential(
    [
        Embedding(input_dim=1000, output_dim=16),  # convierte palabras en vectores
        GlobalAveragePooling1D(),  # reduce cada secuencia a un vector promedio
        Dense(16, activation="relu"),  # capa oculta
        Dense(1, activation="sigmoid"),  # salida binaria (0 = real, 1 = fake)
    ]
)

model.compile(
    loss="binary_crossentropy",  # función de error para clasificación binaria
    optimizer="adam",  # optimizador eficiente
    metrics=["accuracy"],  # queremos medir precisión
)

# ✅ 7. Definimos callbacks para detener el entrenamiento y ajustar la tasa de aprendizaje
early_stop = EarlyStopping(
    monitor="val_loss",  # observamos la pérdida en validación
    patience=2,  # esperamos 2 épocas sin mejora antes de detener
    restore_best_weights=True,  # recuperamos los mejores pesos
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # también observa la pérdida
    factor=0.5,  # reduce la tasa de aprendizaje a la mitad
    patience=1,  # lo hace si no mejora tras 1 época
    verbose=1,  # imprime mensaje en consola
)

# ✅ 8. Entrenamos el modelo con los callbacks definidos
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,
    callbacks=[early_stop, reduce_lr],
    verbose=1,
)

# ✅ 9. Visualizamos el entrenamiento con Seaborn
# Convertimos el historial en un DataFrame para graficar fácilmente
history_df = pd.DataFrame(history.history)

# Creamos una columna para las épocas (índices)
history_df["epoch"] = history_df.index + 1

# 📈 Gráfico de la precisión (accuracy)
plt.figure(figsize=(10, 5))
sns.lineplot(data=history_df, x="epoch", y="accuracy", label="Entrenamiento")
sns.lineplot(data=history_df, x="epoch", y="val_accuracy", label="Validación")
plt.title("Precisión durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Precisión")
plt.legend()
plt.grid(True)
plt.show()

# 📉 Gráfico de la pérdida (loss)
plt.figure(figsize=(10, 5))
sns.lineplot(data=history_df, x="epoch", y="loss", label="Entrenamiento")
sns.lineplot(data=history_df, x="epoch", y="val_loss", label="Validación")
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True)
plt.show()
