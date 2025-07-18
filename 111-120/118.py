# 🧪 Ejercicio 118/200 — Optimización de entrenamiento con callbacks: EarlyStopping y ReduceLROnPlateau
# 1. 📚 Importar librerías necesarias
import re  # Para limpiar texto con expresiones regulares
import pandas as pd  # Para manejar datos en formato tabla
import tensorflow as tf  # Librería de deep learning
import seaborn as sns  # Para gráficos bonitos
import matplotlib.pyplot as plt  # Para mostrar gráficos

from keras.models import Sequential  # Tipo de modelo secuencial (capas una tras otra)
from keras.layers import TextVectorization, Embedding, GlobalAveragePooling1D, Dense
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
)  # Para detener o ajustar el entrenamiento automáticamente
from sklearn.model_selection import (
    train_test_split,
)  # Divide datos en entrenamiento y prueba

# 2. 📰 Simulamos un pequeño conjunto de datos con textos reales y falsos
data = {
    "text": [
        "New education policy announced by government",
        "Aliens invade city center!",
        "WHO warns about new flu variant",
        "Cure for cancer discovered in mushrooms!",
        "President gives speech at UN assembly",
        "Vaccine causes invisibility in rare cases",
    ],
    "label": [0, 1, 0, 1, 0, 1],  # 0 = Noticia real, 1 = Noticia falsa
}
df = pd.DataFrame(data)


# 3. 🧹 Función para limpiar texto (eliminar mayúsculas, URLs y símbolos)
def clean_text(text):
    text = text.lower()  # Convierte todo el texto a minúsculas
    text = re.sub(r"http\S+|www\S+", "", text)  # Elimina enlaces
    text = re.sub(r"[^a-z\s]", "", text)  # Elimina números y símbolos
    return text.strip()  # Elimina espacios innecesarios al inicio y fin


# Aplicamos la función de limpieza a todos los textos
df["text"] = df["text"].apply(clean_text)

# 4. 🔀 División en conjunto de entrenamiento y prueba (67%-33%)
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.33, random_state=42
)

# 5. 🔡 Vectorización del texto: convierte palabras en números
vectorizer = TextVectorization(
    max_tokens=1000,  # Máximo número de palabras a considerar
    output_mode="int",  # Salida como enteros
    output_sequence_length=20,  # Longitud fija de cada texto
)
vectorizer.adapt(X_train)  # Aprende el vocabulario del texto de entrenamiento

# 6. ⚙️ Preparación eficiente de datos con tf.data
train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))  # Crea dataset
train_ds = (
    train_ds.batch(2).cache().prefetch(tf.data.AUTOTUNE)
)  # Agrupa, guarda en caché y anticipa cargas

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(2).cache().prefetch(tf.data.AUTOTUNE)

# 7. 🧠 Modelo de red neuronal simple (rápido y eficiente para CPU)
model = Sequential(
    [
        vectorizer,  # Capa que convierte texto en números
        Embedding(
            input_dim=1000, output_dim=16
        ),  # Convierte números en vectores útiles
        GlobalAveragePooling1D(),  # Promedia la información de todo el texto
        Dense(1, activation="sigmoid"),  # Capa final: da una probabilidad (entre 0 y 1)
    ]
)

# Compilamos el modelo: define cómo aprenderá
model.compile(
    optimizer="adam",  # Algoritmo de optimización
    loss="binary_crossentropy",  # Tipo de error para clasificación binaria
    metrics=["accuracy"],  # Queremos medir precisión
)

# 8. 🛑 Callbacks para mejorar el entrenamiento
early_stop = EarlyStopping(
    monitor="val_loss",  # Observa el error de validación
    patience=2,  # Si no mejora en 2 épocas, se detiene
    restore_best_weights=True,  # Recupera los mejores pesos
    verbose=1,  # Muestra mensaje al detenerse
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",  # Observa el error de validación
    factor=0.5,  # Reduce la tasa de aprendizaje a la mitad
    patience=1,  # Si no mejora en 1 época
    verbose=1,  # Muestra mensaje cuando reduce
)

# 9. 🏋️ Entrenamiento del modelo con los callbacks
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=20,  # Hasta 20 vueltas de entrenamiento
    callbacks=[early_stop, reduce_lr],  # Usa los dos callbacks
    verbose=1,  # Muestra el progreso en consola
)

# 10. 📊 Visualización de métricas del entrenamiento
history_df = pd.DataFrame(history.history)  # Convertimos a tabla para graficar

# 🎯 Gráfico de pérdida (loss)
plt.figure(figsize=(10, 5))
sns.lineplot(data=history_df[["loss", "val_loss"]], linewidth=2.5)
plt.title("Pérdida de Entrenamiento vs Validación", fontsize=14)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Pérdida", fontsize=12)
plt.legend(["Entrenamiento", "Validación"])
plt.grid(True)
plt.tight_layout()
plt.show()

# 🎯 Gráfico de precisión (accuracy)
plt.figure(figsize=(10, 5))
sns.lineplot(
    data=history_df[["accuracy", "val_accuracy"]], linewidth=2.5, palette="muted"
)
plt.title("Precisión de Entrenamiento vs Validación", fontsize=14)
plt.xlabel("Época", fontsize=12)
plt.ylabel("Precisión", fontsize=12)
plt.legend(["Entrenamiento", "Validación"])
plt.grid(True)
plt.tight_layout()
plt.show()
