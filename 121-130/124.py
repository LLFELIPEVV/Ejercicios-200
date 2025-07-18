# 🧠 Ejercicio 124/200 – Entrenamiento de modelo con clases desbalanceadas usando class_weight en Keras

# 📌 Paso 1: Importar librerías necesarias
# Estas son herramientas esenciales para procesamiento, entrenamiento y visualización
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Model
from keras.layers import (
    TextVectorization,
    Embedding,
    GlobalAveragePooling1D,
    Dense,
    Input,
)
from keras.callbacks import EarlyStopping

# 📌 Paso 2: Crear un pequeño conjunto de datos simulado con noticias reales y falsas
# Este dataset tiene más ejemplos de "real" que de "fake", por eso se considera desbalanceado
data = {
    "text": [
        "Breaking news: something real happened!",
        "Shocking! Click here to know the truth",
        "Government confirms the report",
        "Aliens landed in Canada",
        "Study reveals health benefits of green tea",
        "Fake news: NASA hides discovery",
        "Real news: Scientists confirm theory",
        "BREAKING: hoax alert!",
        "Real: President gives new speech",
        "Hoax: cure for cancer found in bananas",
        "Legitimate report from WHO",
        "This is a fake headline with lies",
        "The economy is stable",
        "FALSE: aliens rule Earth",
        "Confirmed: rainfall helps crops",
    ],
    "label": [
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
        "fake",
        "real",
    ],
}
df = pd.DataFrame(data)

# 📌 Paso 3: Visualizar si las clases están balanceadas
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="label", palette="Set2")
plt.title("Distribución de Clases (Real vs Fake)")
plt.xlabel("Etiqueta")
plt.ylabel("Frecuencia")
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.show()


# 📌 Paso 4: Limpiar el texto usando expresiones regulares
# Esto elimina URLs, signos de puntuación y convierte todo a minúsculas para evitar confusión entre palabras
def limpiar_texto(texto):
    texto = texto.lower()  # convertir a minúsculas
    texto = re.sub(r"http\S+", "", texto)  # eliminar URLs
    texto = re.sub(r"[^a-z\s]", "", texto)  # eliminar puntuación
    texto = re.sub(r"\s+", " ", texto).strip()  # eliminar espacios repetidos
    return texto


df["clean_text"] = df["text"].apply(limpiar_texto)

# 📌 Paso 5: Convertir etiquetas 'fake' y 'real' a valores numéricos (0 = fake, 1 = real)
df["label_int"] = df["label"].map({"fake": 0, "real": 1})

# 📌 Paso 6: Separar los datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label_int"],
    test_size=0.2,
    random_state=42,
    stratify=df["label_int"],  # mantener proporción de clases
)

# 📌 Paso 7: Crear un vectorizador de texto con TextVectorization
# Convierte cada frase en una secuencia de números, donde cada número representa una palabra
vectorizador = TextVectorization(
    max_tokens=1000,  # límite de palabras que reconocerá
    output_sequence_length=20,  # todas las secuencias tendrán 20 palabras
    standardize=None,  # ya limpiamos el texto
)

# Adaptamos el vectorizador solo con los textos de entrenamiento
vectorizador.adapt(X_train)

# 📌 Paso 8: Crear datasets optimizados para entrenamiento en TensorFlow
AUTOTUNE = tf.data.AUTOTUNE
batch_size = 4  # cantidad de ejemplos procesados en cada paso

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train))
ds_train = ds_train.batch(batch_size).prefetch(AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
ds_test = ds_test.batch(batch_size).prefetch(AUTOTUNE)

# 📌 Paso 9: Calcular los pesos para cada clase con el objetivo de compensar el desbalance
# Esto hace que el modelo preste más atención a la clase menos frecuente (fake)
pesos = compute_class_weight(
    class_weight="balanced", classes=np.unique(y_train), y=y_train
)
class_weight_dict = dict(enumerate(pesos))
print("Pesos asignados a cada clase:", class_weight_dict)

# 📌 Paso 10: Definir un modelo pequeño y eficiente que funcione bien en CPU
input_layer = Input(shape=(1,), dtype=tf.string)  # entrada: texto sin procesar
x = vectorizador(input_layer)  # paso 1: convertir texto en números
x = Embedding(input_dim=1000, output_dim=16)(x)  # paso 2: mapear palabras a vectores
x = GlobalAveragePooling1D()(x)  # paso 3: resumir información
output = Dense(1, activation="sigmoid")(
    x
)  # paso 4: capa final para clasificación binaria

modelo = Model(input_layer, output)
modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 📌 Paso 11: EarlyStopping – detiene el entrenamiento si no hay mejoras por 3 épocas seguidas
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# 📌 Paso 12: Entrenar el modelo utilizando class_weight para corregir el desbalance
historial = modelo.fit(
    ds_train,
    validation_data=ds_test,
    epochs=20,
    callbacks=[early_stop],
    class_weight=class_weight_dict,
    verbose=1,
)

# 📌 Paso 13: Graficar desempeño durante el entrenamiento
plt.figure(figsize=(10, 4))

# Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(historial.history["loss"], label="Pérdida Entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida Validación")
plt.title("Evolución de la Pérdida")
plt.xlabel("Épocas")
plt.ylabel("Pérdida")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

# Gráfico de exactitud
plt.subplot(1, 2, 2)
plt.plot(historial.history["accuracy"], label="Exactitud Entrenamiento")
plt.plot(historial.history["val_accuracy"], label="Exactitud Validación")
plt.title("Evolución de la Exactitud")
plt.xlabel("Épocas")
plt.ylabel("Exactitud")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()
plt.show()
